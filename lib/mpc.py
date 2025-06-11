import do_mpc
import numpy as np
import casadi as ca
import json


class MPC:
    T_STEP = 0.1  # seconds
    N_HORIZON = 20  # number of steps in the horizon
    N_ROBUST = 0  # number of robust steps (for robust MPC, not used here)

    def __init__(self, filepath="./data/out/fourws_one_side_path.json"):
        self.rotational_inertia = 1000.
        self.mass = 1000.
        self.width = 2.3
        self.engine_power = 1000.

        self.read_path(filepath)
        S, K = self.calculate_curvature()
        self.k = ca.interpolant("curvature", "linear", [S], K)

        values = [self.x, self.y]
        values_flat = np.column_stack((self.x, self.y)).ravel(order="F")
        self.p_interp = ca.interpolant("p", "bspline", [S], values_flat, {})
        
        

        self.model = self.create_model()
        self.model.setup()

        self.controller = do_mpc.controller.MPC(self.model)
        self.controller._settings.store_full_solution = False
        self.controller._settings.t_step = MPC.T_STEP
        self.controller._settings.n_horizon = MPC.N_HORIZON
        self.controller._settings.n_robust = MPC.N_ROBUST
        self.controller._settings.nlpsol_opts['ipopt.max_iter'] = 1000
        self.controller._settings.nlpsol_opts['ipopt.print_level'] = 0
        self.controller._settings.supress_ipopt_output()
        self.set_constraints()
        self.set_objective()

        self.controller._check_validity()
        self.controller.setup()

        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator._settings.t_step = MPC.T_STEP
        self.simulator.setup()

        self.estimator = do_mpc.estimator.StateFeedback(self.model)

    def read_path(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.x = data["x"]
        self.y = data["y"]
        self.length = data["length"]
        self.control_fr = data["control_fr"]
        self.control_rear = data["control_rear"]
        
    def calculate_curvature(self):
        N = len(self.x)
        delta = self.length / N

        S = []  # list of distances from beginning to the point
        K = []  # list of curvatures in all points
        s : float = delta / 2  # start at the middle of the first segment
        for i, control_f, control_r in zip(range(N-1),self.control_fr, self.control_rear):
            angle = control_f #- control_r
            s += delta
            if angle < 0.1: 
                k = 0
            else:
                k = np.tan(angle) / 2.7   # TODO: find the correct equation for R and then K
            if i == 0:
                S.append(0.)
                K.append(k)
            S.append(s)
            K.append(k)
        return S, K
        
    def sdot(self):
        x = self.model.x
        sdot = x['velocity']*ca.cos(x['steering_angle']/2 - x['mu']) / (1 - x['n'] * self.k(x['s']))
        return sdot

    def create_model(self) -> do_mpc.model.Model:
        """
        Setups all variables, inputs, parameters of an MPC model.
        """

        model_type = 'continuous'
        model = do_mpc.model.Model(model_type, 'MX')

        s = model.set_variable("_x", 's', shape=(1,1))
        n = model.set_variable("_x", 'n', shape=(1,1))
        mu = model.set_variable("_x", 'mu', shape=(1,1))
        v = model.set_variable('_x', 'velocity', shape=(1,1))
        steering_angle = model.set_variable('_x', 'steering_angle', shape=(1,1))


        steering_angle_change = model.set_variable('_u', 'steering_angle_change', shape=(1,1))
        acceleration = model.set_variable('_u', 'acceleration', shape=(1,1))

        sdot = v*ca.cos(steering_angle/2 - mu) / (1 - n * self.k(s))

        model.set_rhs('s', sdot)
        model.set_rhs('n', v*ca.sin(steering_angle/2 - mu))
        model.set_rhs('mu',v*ca.tan(steering_angle)/2.7 - self.k(s)*sdot)
        model.set_rhs('velocity', acceleration)
        model.set_rhs('steering_angle', steering_angle_change)

        return model

    def set_objective(
            self,
            control_costs=np.array([[0.1], [0.1]]),
            q_n = 1.0,
            q_mu = 1.0,
            q_B = 1.0
        ):
        
        # Control costs
        u_dim = len(self.model.u.labels())
        assert (control_costs.shape == (u_dim, 1))
        r_term = {key : cost for key, cost in zip(self.model.u.keys(), control_costs)}
        self.controller.set_rterm(**r_term) 

        # State costs
        sdot = self.sdot()
        s = self.model.x['s']
        n = self.model.x['n']
        mu = self.model.x['mu']

        # vref = self.model.track.velocities_interp(s)
        mterm = q_n*(n**2)
        lterm = mterm + (sdot - 1)**2
        self.controller.set_objective(lterm=lterm, mterm=mterm)

    def set_constraints(self):

        # s = self.model.x['s']
        # n = self.model.x['n']
        # mu = self.model.x['mu']
        # self.mpc.set_nl_cons('obstacle_dist_cons', self.obstacle_dist_interp(s, n, mu), 0.)

        TAU = 2*np.pi

        self.controller.bounds['lower', '_x', 's'] = 0.
        self.controller.bounds['lower', '_x', 'mu'] = -TAU/4
        self.controller.bounds['upper', '_x', 'mu'] = TAU/4 

        self.controller.bounds['lower', '_x', 'steering_angle'] = -TAU/8
        self.controller.bounds['upper', '_x', 'steering_angle'] = TAU/8

        self.controller.bounds['lower', '_x', 'velocity'] = 0
        self.controller.bounds['upper', '_x', 'velocity'] = 1

        # INPUT
        self.controller.bounds['lower', '_u', 'steering_angle_change'] = -TAU/4
        self.controller.bounds['lower', '_u', 'acceleration'] = -1 

        self.controller.bounds['upper', '_u', 'steering_angle_change'] = TAU/4
        self.controller.bounds['upper', '_u', 'acceleration'] = 1



