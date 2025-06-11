import do_mpc
import numpy as np
import casadi as ca
import json


class MPCModel:
    def __init__(self):
        self.rotational_inertia = 1000.
        self.mass = 1000.
        self.length_f = 1.5
        self.length_r = 1.5
        self.width = 2.3
        self.engine_power = 1000.

        self.read_path()
        S, K = self.calculate_curvature()
        self.k = ca.interpolant(
            "curvature", "linear", [S],
            K
        )
        self.model = self.create_model()
        self.model.setup()
        self.model = model
        self.mpc_model = self.model.model
        self.t_step = t_step
        self.mpc = do_mpc.controller.MPC(self.mpc_model)
        self.mpc._settings.store_full_solution = False
        self.mpc._settings.t_step = t_step
        self.mpc._settings.n_horizon = n_horizon
        self.mpc._settings.n_robust = n_robust
        self.mpc._settings.nlpsol_opts['ipopt.max_iter'] = 1000
        self.mpc._settings.nlpsol_opts['ipopt.print_level'] = 0
        # self.mpc._settings.supress_ipopt_output()

        # rho determines the shape of the friction ellipse constraint
        # alpha determines the maximum combined force. (the size of the ellipse)
        alpha, rho = 1.0, 1.0

        # q_n is a cost for deviating from the optimal path (in perpendicular to path direction)
        # q_mu is a cost for a heading that deviates from the optimal path
        # q_B penalizes the difference between the kinematic and dynamic side slip angle.
        q_n, q_mu, q_B = 0.5, 3.0, 1e-2
        self.set_constraints(rho, alpha)
        self.set_objective(control_costs, q_n, q_mu, q_B)

        self.mpc._check_validity()
        self.mpc.setup()

    def read_path(self):
        path_file = "./data/out/fourws_one_side_path.json"  # lub inna ścieżka do pliku
        with open(path_file, "r") as f:
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
        s = delta / 2  # start at the middle of the first segment
        for i, control_f, control_r in zip(range(N-1),self.control_fr, self.control_rear):
            angle = control_f #- control_r
            s += delta
            if angle < 0.1: 
                k = 0
            else:
                k = np.tan(angle) / 2.7   # TODO: find the correct equation for R and then K
            if i == 0:
                S.append(0)
                K.append(k)
            S.append(s)
            K.append(k)
        return S, K
        
    def sdot(self):
        x = self.model.x
        sdot = (x['vx']*ca.cos(x['mu']) - x['vy']*ca.sin(x['mu'])) / (1 - x['n'] * self.k(x['s']))
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

        vx = model.set_variable('_x', 'vx', shape=(1,1))
        vy = model.set_variable('_x', 'vy', shape=(1,1))
        r = model.set_variable('_x', 'r', shape=(1,1))

        steering_angle = model.set_variable('_x', 'steering_angle', shape=(1,1))
        throttle = model.set_variable('_x', 'throttle', shape=(1,1))

        steering_angle_change = model.set_variable('_u', 'steering_angle_change', shape=(1,1))
        throttle_change = model.set_variable('_u', 'throttle_change', shape=(1,1))

        sdot = (vx*ca.cos(mu) - vy*ca.sin(mu)) / (1 - n * self.k(s))

        model.set_rhs('s', sdot)
        model.set_rhs('n', 
            vx*ca.sin(mu) + vy*ca.cos(mu)
        )
        model.set_rhs('mu',
            r - self.k(s)*sdot                    
        )

        model.set_rhs('throttle', throttle_change)
        model.set_rhs('steering_angle', steering_angle_change)
        
        return model
    
    def set_objective(self, control_costs, q_n = 1.0, q_mu = 1.0, q_B = 1.0):
        u_dim = len(self.mpc_model.u.labels())
        assert (control_costs.shape == (u_dim, 1))
        
        r_term = {key : cost for key, cost in zip(self.mpc_model.u.keys(), control_costs)}
        self.mpc.set_rterm(**r_term) 

        # Get variables from model
        sdot = self.model.sdot()
        s = self.mpc_model.x['s']
        n = self.mpc_model.x['n']
        mu = self.mpc_model.x['mu']
        vx = self.mpc_model.x['vx']
        vy = self.mpc_model.x['vy']

        vref = self.model.track.velocities_interp(s)
        mterm = q_n*(n**2) + q_mu*(mu**2) + vy**2
        lterm = mterm + (vx - 0.6*vref)**2 + self.model.B(q_B)

        self.mpc.set_objective(lterm=lterm, mterm=mterm)

    def set_constraints(self, rho, alpha):
        s = self.mpc_model.x['s']
        n = self.mpc_model.x['n']
        mu = self.mpc_model.x['mu']
        vx = self.mpc_model.x['vx']
        vy = self.mpc_model.x['vy']
        r = self.mpc_model.x['r']
        steering_angle = self.mpc_model.x['steering_angle']
        throttle = self.mpc_model.x['throttle']

        # non-linear constraints (for the vehicle to stay in track)
        left, right = self.model.get_lateral_constraint(s, n, mu)
        self.mpc.set_nl_cons('left_dist_cons', left, 0.)
        self.mpc.set_nl_cons('right_dist_cons', right, 0.)

        # front, back = self.model.get_traction_ellipse_constraint(throttle, vx, vy, r, steering_angle, rho, alpha)
        # self.mpc.set_nl_cons('front_traction_ellipse_cons', front, 0., soft_constraint=True)
        # self.mpc.set_nl_cons('back_traction_ellipse_cons', back, 0., soft_constraint=True)

        # TODO Set constraints for velocities, from calculated optimal max_velocities. 
        not_set = 0.
        # Lower state bounds
        self.mpc.bounds['lower', '_x', 's'] = 0.
        self.mpc.bounds['lower', '_x', 'mu'] = -np.pi*0.5
        self.mpc.bounds['lower', '_x', 'vx'] = 0. 
        # self.mpc.bounds['lower', '_x', 'vy'] = not_set
        # self.mpc.bounds['lower', '_x', 'r'] = not_set
        self.mpc.bounds['lower', '_x', 'steering_angle'] = -np.pi/4
        self.mpc.bounds['lower', '_x', 'throttle'] = -1

        # # Upper state bounds
        # self.mpc.bounds['upper', '_x', 's'] = not_set
        # self.mpc.bounds['upper', '_x', 'n'] = not_set 
        self.mpc.bounds['upper', '_x', 'mu'] = np.pi*0.5 
        # self.mpc.bounds['upper', '_x', 'vx'] = not_set
        # self.mpc.bounds['upper', '_x', 'vy'] = not_set
        # self.mpc.bounds['upper', '_x', 'r'] = not_set 
        self.mpc.bounds['upper', '_x', 'steering_angle'] = np.pi/4
        self.mpc.bounds['upper', '_x', 'throttle'] = 1

        # Lower input bounds
        self.mpc.bounds['lower', '_u', 'steering_angle_change'] = -2*np.pi/4
        self.mpc.bounds['lower', '_u', 'throttle_change'] = -1 

        # Upper input bounds
        self.mpc.bounds['upper', '_u', 'steering_angle_change'] = 2*np.pi/4
        self.mpc.bounds['upper', '_u', 'throttle_change'] = 1



