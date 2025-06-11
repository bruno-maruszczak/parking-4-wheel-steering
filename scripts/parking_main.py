import os
from pathlib import Path
# local files
from lib.parking import  Car, ParkingLotBitMap
from lib.OMPL import OMPL
from lib.plot import Plot
import matplotlib.pyplot as plt

def experimental_plot():
    img = plt.imread("./data/map/parking_layout.png")
    plt.imshow(img)
    plt.show()

def main():

    parking_lot_bitmap = ParkingLotBitMap(Path(os.getcwd()).joinpath("data/map/parking_layout.npy"))
    parking_lot_bitmap.get_obstacles()
    # print(parking_lot_bitmap.obstacles)

    plot_bounds = parking_lot_bitmap.PLOT_BOUNDS
    # Debug plotting
    # parking_lot_bitmap.plot()
    # parking_lot_bitmap.map_plot(plot_bounds)

    car = Car()

    # parking_lot.init_parking_slots(car.CAR_LEN,car.CAR_WID)
    ompl = OMPL(plot_bounds, car, parking_lot_bitmap)
    
    colours = {"bicycle": "deepskyblue", "4ws": "orange"}
    results = {m: ompl.create_planner(m) for m in ("bicycle", "4ws")}

    plot = Plot(parking_lot_bitmap, results, plot_bounds, colours, car)
    plot.static_plot(Path(os.getcwd()).joinpath("data/out/parking_paths.png"))
    # plot.animate(Path(os.getcwd()).joinpath("data/out/parking_animation.gif"))


if __name__ == "__main__":
    main()