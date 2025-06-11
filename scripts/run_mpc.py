import numpy as np
import matplotlib.pyplot as plt
import json
import do_mpc
import casadi as ca
from pathlib import Path
import os

from lib.mpc import MPC

def main(): 
    print("Loading trajectory...")
    planned_path = Path(os.getcwd()).joinpath("data/out/fourws_one_side_path.json")
    with open(planned_path, "r") as f:
        data = json.load(f)
    path_length = data["length"]
    print("Loading vehicle model...")
    mpc = MPC(planned_path)

    print("Creating MPC controller...")
    controller = mpc.controller
    simulator = mpc.simulator
    estimator = mpc.estimator

    print("Initializing state...")
    s0, n0, mu0 = 0., 0., 0.
    v0 = 0.0
    steer_angle0, throttle0 = 0., 0.1
    x0 = np.reshape([s0, n0, mu0, v0, steer_angle0], (-1, 1))
    u0 = np.array([[0.0], [0.0]])

    controller.x0 = x0 
    controller.set_initial_guess()
    simulator.x0 = x0
    estimator.x0 = x0

    print("Initializing traces...")
    steps = 500
    X = np.zeros((steps + 1, *x0.shape))
    Y = np.zeros((steps + 1, *x0.shape))
    U = np.zeros((steps + 1, *u0.shape))
    X[0] = x0
    Y[0] = x0
    U[0] = u0

    print("Running simulation...")
    last_s = x0[0]
    x, u = x0, u0
    last_step = 0
    for i in range(1, steps + 1):
        print(f"\n--- STEP {i:4d} ---")
        u = controller.make_step(x)
        y = simulator.make_step(u)
        x = estimator.make_step(y)
        X[i] = x
        Y[i] = y
        U[i] = u
        if x[0] - path_length > 0.1:
            print("Reached end of path, stopping simulation.")
            break
        print(f"sdot {(x[0] - last_s) / MPC.T_STEP}")
        print(f"s {x[0]}")
        last_s = x[0]
        last_step = i

    with open(Path(os.getcwd()).joinpath('data/out/mpc_results.json'), 'w') as f:
        data = {
            's':        [float(x[0]) for x in X[:last_step]],
            'n':        [float(x[1]) for x in X[:last_step]],
            'mu':       [float(x[2]) for x in X[:last_step]],
            'v':        [float(x[3]) for x in X[:last_step]],
            'steer':    [float(x[4]) for x in X[:last_step]]
        }
        json.dump(data, f)


if __name__ == "__main__":
    main()