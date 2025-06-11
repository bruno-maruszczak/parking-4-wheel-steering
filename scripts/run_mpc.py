import numpy as np
import matplotlib.pyplot as plt
import json
from mpc.track import Track as Track
from mpc.model import VehicleModel
from mpc.controller import Controller
from mpc.simulator import Simulator
from visualiser import Visualiser

import do_mpc

import casadi as ca


def main(): 
    print("Loading trajectory...")
    track = Track("MX-5", "buckmore", method, n_samples)
    path = track.optimal_path


    
    print("Loading vehicle model...")
    model = VehicleModel('./data/vehicles/MX5.json', track)
    print("Creating MPC controller...")
    controller = Controller(model, np.reshape([1e-2, 1e-2], (-1, 1)))

    print("Initializing state...")
    s0, n0, mu0 = 0., 0., 0.
    vx0, vy0, r0 = 5.0, 0.0, 0.
    steer_angle0, throttle0 = 0., 0.1
    x0 = np.reshape([s0, n0, mu0, vx0, vy0, r0, steer_angle0, throttle0], (-1, 1))


    print("Creating simulator...")
    simulator = Simulator(model)
    sim = simulator.simulator
    sim.x0 = x0
    controller.mpc.x0 = x0 
    controller.mpc.set_initial_guess()
    estimator = do_mpc.estimator.StateFeedback(model.model)
    estimator.x0 = x0


    u0 = np.array([[0.0], [0.0]])
    
    steps = 500
    X = np.zeros((steps + 1, *x0.shape))
    X[0] = x0
    Y = np.zeros((steps + 1, *x0.shape))
    Y[0] = x0
    Fys = np.zeros((steps+1, 2))
    alphas = np.zeros((steps+1, 2))
    Fys[0] = [0.0, 0.0]
    alphas[0] = [0.0, 0.0]

    U = np.zeros((steps + 1, *u0.shape))
    U[0] = u0
    last_s = x0[0]
    for i in range(1, steps + 1):
        print(f"\n---------------------------\nsimulation step: {i}\n---------------------------\n")
        u0 = controller.mpc.make_step(x0)
        y = sim.make_step(u0)
        x0 = estimator.make_step(y)
        X[i] = x0
        Y[i] = y
        U[i] = u0
        alpha_f, alpha_r = model.get_slip_angles(x0[3][0], x0[4][0], x0[5][0], x0[6][0])
        alphas[i,:] = np.array([alpha_f, alpha_r])
        Fy_f, Fy_r = model.get_lateral_forces(alpha_f, alpha_r)
        Fys[i,:] = np.array([Fy_f, Fy_r])
        print(f"sdot {(x0[0] - last_s) / controller.t_step}")
        last_s = x0[0]


    with open('sim_results.json', 'w') as f:
        data = {'x': X.tolist(), 'y': Y.tolist(), 'u': U.tolist(), "Fy": Fys.tolist(), "alpha": alphas.tolist()}
        json.dump(data, f)
    
    Visualiser(track, 'sim_results.json')

    fig, ax, sim_graphics = simulator.plot_results()
    sim_graphics.plot_results()
    sim_graphics.reset_axes()

    plt.show()

if __name__ == "__main__":
    main()