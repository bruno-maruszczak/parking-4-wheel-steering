import json
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt



class MPC_plot:
    def __init__(self, p_interp : ca.interpolant, sym : ca.MX.sym):
        s = sym
        self.p_interp = ca.Function('point', [s], [p_interp])
        
        tangent = ca.jacobian(p_interp, s)     # exact dp/ds
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
        x_not_shifted = []; y_not_shifted = []
        x_list = []; y_list = []
        for s, n in zip(self.mpc_results['s'], self.mpc_results['n']):
            p = self.p_interp(s)
            x = float(p[0])
            y = float(p[1])
            x_not_shifted.append(x); y_not_shifted.append(y)

            normal_shift = self.normal_vector_interp(s)*n

            # print(normal_shift)
            x_list.append(float(normal_shift[0]) + x)
            y_list.append(float(normal_shift[1]) + y)

        print(x_not_shifted)

        # Plot x values
        plt.figure()
        plt.plot(x_not_shifted, y_not_shifted, label='Not shifted')
        plt.plot(x_list, y_list, label='Shifted')
        x = self.reference_data['x']
        y = self.reference_data['y']
        plt.plot(x,y, label='Reference')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('MPC Path Comparison')
        plt.show()


    



