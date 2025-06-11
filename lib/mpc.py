#file for mpc planner if was to be implemented

import do_mpc
import re
import casadi as ca
import json

class MPCModel:
    def __init__(self):
        self.optimal_path = self.track.optimal_path


        self.width = 1.0
        self.model = self.create_model()
        self.model.setup()

    def remove_comments(self, json_str):
        # Remove single-line comments
        json_str = re.sub(r'//.*', '', json_str)
        # Remove multi-line comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        return json_str
    
    def load_params(self, path):
        """Load vehicle data from JSON file."""
        with open(path, 'r') as f:
            json_str = f.read()
            json_str_clean = self.remove_comments(json_str)
            data = json.loads(json_str_clean)
            self.rotational_inertia = data["rotational_inertia"]
            self.name = data["name"]
            self.mass = data["mass"]

    def k(self, s):
        return self.optimal_path.find_curvature_at_s(s)

    
    def get_lateral_constraint(self, s, n, mu):
        length = self.length_f + self.length_r
        width = self.width

        NL = self.track.find_dist_to_band_symb(s, "left")
        NR = self.track.find_dist_to_band_symb(s, "right")
        
        left_constraint = n - length * 0.5 * ca.sin(ca.sign(mu) * mu) + width * 0.5 * ca.cos(mu) - NL
        right_constraint = - n + length * 0.5 * ca.sin(ca.sign(mu) * mu) + width * 0.5 * ca.cos(mu) - NR

        # TODO what are these for?
        # left_constraint_trunc = ca.if_else(left_constraint > NL, left_constraint - NL, 0.0)
        # right_constraint_trunc = ca.if_else(right_constraint > NR, right_constraint - NR, 0.0)

        return left_constraint, right_constraint
    
    def get_motor_force(self, throttle):
        return self.C_m * throttle

    def sdot(self):
        x = self.model.x
        sdot = (x['vx']*ca.cos(x['mu']) - x['vy']*ca.sin(x['mu'])) / (1 - x['n'] * self.k(x['s']))
        return sdot

    def B(self, q_B):
        x = self.model.x
        b_dyn = ca.atan(x['vy']/x['vx'])
        b_kin = ca.atan(x['steering_angle']*self.length_r / (self.length_f + self.length_r))
        return q_B * (b_dyn - b_kin)**2

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