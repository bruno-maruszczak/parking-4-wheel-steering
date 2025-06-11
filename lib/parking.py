from shapely.geometry import Polygon
import numpy as np
from shapely.affinity import rotate, translate
import math
import cv2
import matplotlib.pyplot as plt


class Car:
    def __init__(self):
        # Car that will move (our robot)
        self.CAR_LEN = 4.5   # m
        self.CAR_WID = 2.0   # m  (larger relative to slot for realism)

    # ------------------ Helper to make robot car polygon -----------------------
    def polygon(self, x: float, y: float, yaw: float) -> Polygon:
        rect = Polygon([
            (-self.CAR_LEN/2, -self.CAR_WID/2), (self.CAR_LEN/2, -self.CAR_WID/2),
            (self.CAR_LEN/2,  self.CAR_WID/2), (-self.CAR_LEN/2,  self.CAR_WID/2)
        ])
        rect = rotate(rect, math.degrees(yaw), origin=(0, 0), use_radians=False)
        return translate(rect, x, y)

class ParkingLotBitMap:
    def __init__(self, bitmap_path, decorations=None):
        # Wczytaj bitmapę parkingu z pliku numpy
        print(bitmap_path)
        self.bitmap = np.load(bitmap_path)
        self.decorations = np.load(decorations) if decorations else None
        self.scale = 50 # 50 pixels is one meter
        self.width_px = self.bitmap.shape[1]
        self.length_px = self.bitmap.shape[0]
        self.width_m = self.width_px / self.scale
        self.length_m = self.length_px / self.scale

        self.PLOT_BOUNDS =  [0, self.width_m, 0, self.length_m]

        self.lot_boundary = Polygon([
            (self.PLOT_BOUNDS[0], self.PLOT_BOUNDS[2]), (self.PLOT_BOUNDS[1], self.PLOT_BOUNDS[2]),
            (self.PLOT_BOUNDS[1], self.PLOT_BOUNDS[3]), (self.PLOT_BOUNDS[0], self.PLOT_BOUNDS[3])
        ])

    def get_obstacles(self):
        self.obstacles = []
        self.slot_lines = []
        # Add 1-pixel black padding around the binary image
        # self.bitmap = np.pad(self.bitmap, pad_width=1, mode='constant', constant_values=0)

        binary = 1 - self.bitmap

        # Find contours (external only)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)        
        for cnt in contours:
            if cv2.contourArea(cnt) > 0.5 * binary.size:
                continue  # omit the biggest contour (the frame of the bit map)

            # change to Polygon (cv2 contours: (N,1,2) -> (N,2))
            coords = cnt.squeeze()
            # print(coords)
            poly = Polygon([(x/self.scale, y/self.scale) for [x, y] in coords])
            self.obstacles.append(poly)

        if self.decorations is None:
            raise ValueError("Decorations array is None. Please provide a valid decorations file.")
        binary = 1 - self.decorations
        # Find contours (external only)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        for cnt in contours:
            if cv2.contourArea(cnt) > 0.5 * binary.size:
                continue  # omit the biggest contour (the frame of the bit map)

            # change to Polygon (cv2 contours: (N,1,2) -> (N,2))
            coords = cnt.squeeze()
            # print(coords)
            poly = Polygon([(x/self.scale, y/self.scale) for [x, y] in coords])
            
            self.slot_lines.append(poly)


    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.bitmap, cmap="gray", origin="lower")
        ax.set_title("Parking lot bitmap")
        
        # Plot obstacles if they exist
        if hasattr(self, "obstacles"):
            for poly in self.obstacles:
                x, y = poly.exterior.xy
                x *= self.scale; y *= self.scale
                ax.plot(x, y, color="red", linewidth=1)

        plt.show()

    def map_plot(self, plot_bounds):
         
        plt.figure(figsize=(9, 12))
        ax = plt.gca(); ax.set_aspect('equal', 'box')
        ax.set_xlim(plot_bounds[0], plot_bounds[1])
        ax.set_ylim(plot_bounds[2], plot_bounds[3])
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(False)

        # for slot in self.parking_lot.slot_lines:
        #     x, y = slot.exterior.xy
        #     ax.plot(x, y, color="#aaaaaa", linewidth=1)

        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.plot(x, y, color="#902626")
        
        plt.show()


