import json
import numpy as np
import matplotlib.pyplot as plt
import os
import casadi as ca
from scipy.interpolate import splprep, splev


class Track:
    def __init__(self, vehicle_name, track_name, method_name, n_samples):
        cwd = os.getcwd()
        base_path = os.path.join(cwd, "data", "plots", vehicle_name, track_name, method_name)

        self.left_bound_x, self.left_bound_y = self.load_path_from_json(os.path.join(base_path, "left.json"))
        self.right_bound_x, self.right_bound_y = self.load_path_from_json(os.path.join(base_path, "right.json"))
        self.path_x, self.path_y = self.load_path_from_json(os.path.join(base_path, "path.json"))
        self.widths = self.load_path_from_json(os.path.join(base_path, "widths.json"))
        self.velocities = self.load_path_from_json(os.path.join(base_path, "velocities.json"))
        self.n_samples = n_samples
        self.left_bound = ControllerReferencePath(np.array([self.left_bound_x, self.left_bound_y]), closed=True, n_samples=self.n_samples)
        self.right_bound = ControllerReferencePath(np.array([self.right_bound_x, self.right_bound_y]), closed=True, n_samples=self.n_samples)
        self.optimal_path = ControllerReferencePath(np.array([self.path_x, self.path_y]), closed=True, n_samples=self.n_samples)

        # Create distance lookup tables, for distance form bands
        self.bound_dist_table = {
            "left": self.create_distance_table(side="left"), 
            "right": self.create_distance_table(side="right")
        }
        arc = self.optimal_path.arc_lengths_sampled
        self.bound_dist_interp = {
            side: ca.interpolant(
                f"dist_{side}", "linear", [arc],
                np.array(self.bound_dist_table[side])
            )
            for side in ["left", "right"]
        }
        # Create velocity lookup table
        self.velocities_interp = ca.interpolant(
            "velocities", "linear", [arc],
            np.array(self.velocities)
        )

    def load_path_from_json(self, filepath):
        """Load path data from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        name = data["name"]
        if name == "widths":
            return np.array(data["width"])
        elif name == "velocities":
            return np.array(data["velocities"])
        else:
            x = np.array(data["path"]["x"])
            y = np.array(data["path"]["y"])
            return x, y
    
    def plot(self):
        plt.scatter(self.left_bound_x, self.left_bound_y, color='tab:blue', marker='.')
        plt.scatter(self.right_bound_x, self.right_bound_y, color='tab:orange', marker='.')
        plt.scatter(self.path_x, self.path_y, color='tab:green', marker='.')

    def plotting_spline(self, spline):
        """Plot the spline."""
        x_fine, y_fine = splev(spline.u_sampled, spline.spline)
        plt.plot(x_fine, y_fine, 'tab:red')

    def find_tangent_line(self, x0, y0, dx, dy):
        """
        Finds a line tangent to a curve at a given point
        
        Parameters:
        x0, y0 - the given point
        dx, dy - derivatives of a curve in respect to the parameter u

        Returns:
        - a tuple (A, B, C) in general line form: Ax + By + C = 0
        """
        if dx == 0:
            return (0, 1, -x0)
        else:
            slope = dy/dx
            return (-1, slope, y0-slope*x0)
    
    def find_perpendicular_line(self, line : tuple, x0, y0):
        """
        Finds a line perpendicular to a line at a given point 
        
        Parameters:
        x0, y0 - the given point
        line - a tuple (A, B, C) in general line form: Ax + By + C = 0

        Returns:
        - a tuple (A, B, C) in general line form: Ax + By + C = 0 of the perpendicular line
        """
        A, B, C = line
        assert A != 0 or B != 0
        if A == 0:
            perp_line = (0, 1, -y0)
        elif B == 0:
            perp_line = (1, 0, -x0)
        else:
            slope = B / A
            perp_line = (-slope, 1, -y0 + slope*x0)
        return perp_line
    
    def find_dist_to_band_symb(self, s, side : str="left"):  
        assert side in ["left", "right"]
        if isinstance(s, ca.MX):
            return self.bound_dist_interp[side](s)

    def find_dist_to_band(self, s, side : str="left"):
        """
        Find distance between optimal trajectory and a given band (left/right)

        Parameters:
        s - the given point on a curve (as arc-length), if s is an casadi expression, returns an expression for distance.
        side - ["left", "right"] to choose the band

        Returns:
        - distance to a chosen band
        """
        assert side in ["left", "right"]
        
        u = self.optimal_path.find_u_given_s(s)
        # Find x, y, dx, dy of the spline
        tck = self.optimal_path.spline
        x0, y0 = splev(u, tck)
        dx, dy = splev(u, tck, der=1)
        # find perpendicular_line
        tangent = self.find_tangent_line(x0, y0, dx, dy)
        perpendicular_line = self.find_perpendicular_line(tangent, x0, y0)
        A, B, C = perpendicular_line
        # Get sampled points on a given band
        bound = self.left_bound if side == "left" else self.right_bound
        x, y = bound.get_sample_points()

        # Sort points by the distance from line
        distances = np.abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
        sorted_indices = np.argsort(distances)

        # Get the two closest points
        closest_points = [(x[i], y[i]) for i in sorted_indices[:]]

        # Choose the one closer to the (x0, y0) point on the curve
        width = 10
        closest_point = None
        min_distance = float('inf')
        for point in closest_points:
            dist = np.hypot(point[0] - x0, point[1] - y0)
            if dist < min_distance and dist <= width: 
                min_distance = dist
                closest_point = point

        if closest_point is None:
            raise ValueError("No point found within the radius of {width}")

        distance = min_distance

        return distance

    def create_distance_table(self, side : str="left"):
        """
        Creates lookup tables for distance from optimal path to left and right band
        """
        assert side in ["left", "right"]

        return [self.find_dist_to_band(s, side=side) for s in self.optimal_path.arc_lengths_sampled]


import numpy as np
from scipy.interpolate import splev, splprep
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

import casadi as ca

from typing import List, Tuple


def cumulative_distances(points):
    """Returns the cumulative linear distance at each point."""
    d = np.cumsum(np.linalg.norm(np.diff(points, axis=1), axis=0))
    return np.append(0, d)


class Path:
    """Wrapper for scipy.interpolate.BSpline."""

    def __init__(self, controls, closed):
        """Construct a spline through the given control points."""
        self.controls = controls
        self.closed = closed
        self.dists = cumulative_distances(controls)
        self.spline, _ = splprep(controls, u=self.dists, k=3, s=0, per=self.closed)
        self.length = self.dists[-1]


    def position(self, s=None):
        """Returns x-y coordinates of sample points."""
        if s is None:
            return self.controls
        x, y = splev(s, self.spline)
        return np.array([x, y])

    def curvature(self, u=None, return_absolute_value=True):
        """
        Calculate curvature of spline at given parameter u value
        
        Parameters:
        - u: parameter values at which to compute the curvature
        
        Returns:
        - curvature: the curvature values at each parameter value u
        """
        
        if u is None:
            u = self.dists

        # First derivatives dx/du and dy/du
        dx_du, dy_du = splev(u, self.spline, der=1)
        
        # Second derivatives d2x/du2 and d2y/du2
        d2x_du2, d2y_du2 = splev(u, self.spline, der=2)
        
        # Curvature formula: kappa(u) = |dx/du * d2y/du2 - dy/du * d2x/du2| / (dx/du^2 + dy/du^2)^(3/2)

        curvature = (dx_du * d2y_du2 - dy_du * d2x_du2) / (dx_du**2 + dy_du**2)**(3/2)

        
        return np.abs(curvature) if return_absolute_value else curvature
    
    def gamma2(self, u=None):
        """Returns the sum of the squares of sample curvatures, Gamma^2."""
        if u is None:
            u = self.dists

        # First derivatives dx/du and dy/du
        dx_du, dy_du = splev(u, self.spline, der=1)
        
        # Second derivatives d2x/du2 and d2y/du2
        d2x_du2, d2y_du2 = splev(u, self.spline, der=2)
        
        # Curvature formula: kappa(u) = |dx/du * d2y/du2 - dy/du * d2x/du2| / (dx/du^2 + dy/du^2)^(3/2)
        curvature = (dx_du * d2y_du2 - dy_du * d2x_du2) / (dx_du**2 + dy_du**2)**(3/2)
        curvature = curvature**2
        return np.sum(curvature)






class ControllerReferencePath(Path):
    """Wrapper for scipy.interpolate.BSpline."""

    def __init__(self, controls, closed, n_samples=1000):
        super().__init__(controls, closed)

        # sample u, to calculate transformation u -> arc_length (return u given arc length)
        self.u_sampled = np.linspace(0, self.length, n_samples)
        self.arc_lengths_sampled = self.calculate_arc_length()

        #Generate curvature(s) as a lookup table, where s - arc legnth, 
        # for when s is a symbolic variable and its numerical value is unknown, during call
        self.curvature_lookup_table = self.create_curvature_table(n_samples)
        s_vals, k_vals = zip(*self.curvature_lookup_table)
        self.curvature_interp = ca.interpolant(
            "curvature_interp", "linear",
            [np.array(s_vals)], np.array(k_vals)
        )

    @classmethod
    def fromPath(cls, path : Path, n_samples=1000):
        return cls(path.controls, path.closed, n_samples)

        
    def piecewise_linear_interpolation(self,x, x_values, y_values):
        """
        Function should return an casadi expression that represents piecewise linear interpolation of a given lookup-table
        """
        # Create a CasADi MX variable for the result
        result = ca.MX(0.)
        
        # Iterate over the intervals between x_values
        for i in range(len(x_values) - 1):
            # Get the current and next x and y values
            x0, x1 = x_values[i], x_values[i + 1]
            y0, y1 = y_values[i], y_values[i + 1]
            
            # Linear interpolation between (x0, y0) and (x1, y1)
            slope = (y1 - y0) / (x1 - x0)
            interpolation = y0 + slope * (x - x0)
            
            # Apply this interpolation if x is between x0 and x1
            result = ca.if_else(ca.logic_and(x >= x0, x < x1), interpolation, result)
        
        # Handle the case when x is exactly equal to the last x_value
        result = ca.if_else(x == x_values[-1], y_values[-1], result)
        return result

    def create_curvature_table(self, n_samples: int) -> List[Tuple[float, float]]:
        """
        Creates a lookup table for curvature given an arc-length s for current track.
        This is needed when arc-length is a casadi symbolic expression, so then curvature function is defined as a piecewise interpolation of the lookup table.

        Parameters:
        - n_samples: number of arc-length samples used to create the table

        - Returns: a list of tuples where first is the argument and second is the function value for y = f(x) -> [(x0, y0), (x1, y1), ...]
        """
        s_max = self.arc_lengths_sampled[-1]
        s_values = np.linspace(0, s_max, n_samples)

        def find_curvature(s):
                    # Interpolate to find the corresponding u for the given arc length s
            u = self.find_u_given_s(s)
        
            # Calculate curvature at the interpolated u value
            curvature = self.curvature(u, return_absolute_value=False)
        
            return curvature
        table = [(s, find_curvature(s)) for s in s_values]
        return table

    def calculate_arc_length(self):
        """
        Calculate the cumulative arc length for each parameter value u.

        Returns:
        - arc_lengths: array of arc length values corresponding to u_fine
        """
        # Evaluate first derivatives dx/du and dy/du
        dx_du, dy_du = splev(self.u_sampled, self.spline, der=1)
        
        # Calculate differential arc length: ds = sqrt((dx/du)^2 + (dy/du)^2)
        ds_du = np.sqrt(dx_du**2 + dy_du**2)
        
        # Compute cumulative arc length by integrating ds/du
        arc_lengths = cumulative_trapezoid(ds_du, self.u_sampled, initial=0)
        
        return arc_lengths

    def find_u_given_s(self, s):
        """
        Find the paramter u of curve that represents given travelled arc lengths (from beginning)

        Parameters:
        - s : desired arc lengths (distance from beginning of the curve)

        Returns:
        - u : parametrs u that represent given arc lengths on the curve
        """
        u = np.interp(s, self.arc_lengths_sampled, self.u_sampled)
        return u

    def find_curvature_at_s(self, s):
        """
        Find the curvature at a given arc length s using the precomputed arc lengths and spline representation.
        
        Parameters:
        - s: desired arc length (distance from beginning of the curve)
            can also be a casadi symbolic expression (casadi.casadi.SX)
    
        Returns:
        - curvature: curvature at the given arc length s
        """

        # if s is a casadi symbolic expression
        # if isinstance(s, ca.MX):
        return self.curvature_interp(s)



    def get_sample_points(self):
        """
        Return the finely sampled points on the curve.
        """
        return splev(self.u_sampled, self.spline)