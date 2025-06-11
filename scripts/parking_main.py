import os
from pathlib import Path
# local files
from lib.parking import  Car, ParkingLotBitMap
from lib.OMPL import OMPL
from lib.plot import Plot
import matplotlib.pyplot as plt

def main():

    parking_lot_bitmap = ParkingLotBitMap(
        Path(os.getcwd()).joinpath("data/map/parking_obstacles.npy"),
        Path(os.getcwd()).joinpath("data/map/parking_decorations.npy"),
        )
    parking_lot_bitmap.get_obstacles()
    plot_bounds = parking_lot_bitmap.PLOT_BOUNDS
    # Debug plotting
    # parking_lot_bitmap.plot()
    # parking_lot_bitmap.map_plot(plot_bounds)

    car = Car()
    ompl = OMPL(plot_bounds, car, parking_lot_bitmap)
    
    colours = {"bicycle": "deepskyblue", "fourws_one_side": "orange", "fourws_two_side": "black"}
    results = {m: ompl.create_planner(m) for m in ("bicycle", "fourws_one_side", "fourws_two_side")}

    plot = Plot(parking_lot_bitmap, results, plot_bounds, colours, car)
    plot.static_plot(Path(os.getcwd()).joinpath("data/out/parking_paths.png"))
    plot.animate(Path(os.getcwd()).joinpath("data/out/parking_animation.gif"), order=["bicycle", "fourws_one_side", "fourws_two_side"])


if __name__ == "__main__":
    main()