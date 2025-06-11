import json
import numpy as np
import casadi as ca


class MPC_plot:
    def __init__(self, p_interp : ca.interpolant):

        self.p_interp = p_interp
        
        s = ca.MX.sym('s')
        tangent = ca.jacobian(p_interp(s), s)     # exact dp/ds
        normal = ca.vertcat(-tangent[1], tangent[0])
        unit_normal = normal / ca.norm_2(normal)
        self.normal_vector_interp = ca.Function('normal', [s], [unit_normal])

        self.read_files()
        self.create_mpc_path()
        


    def read_files(self):

        with open("data/out/mpc_results.json", "r") as f:
            self.mpc_results = json.load(f)

        with open("data/out/fourws_one_side_path.json", "r") as f:
            self.reference_data = json.load(f)

    def create_mpc_path(self):

        x_list = []; y_list = []
        for s, n in zip (self.mpc_results['s'], self.mpc_results['n']) :
            p = self.p_interp(s)
            x = p[0]
            y = p[1]

            normal_shift = self.normal_vector_interp(s)*n

            print(normal_shift)
            x_list.append(normal_shift[0] + x)
            y_list.append(normal_shift[1] + y)

    



