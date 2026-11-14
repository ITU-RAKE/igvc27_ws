"""
ika_utils.costmap_utils.Costmap yerine gecen kucuk sinif (path_planner icin).

data[y, x]: OccupancyGrid verisi (satir = y, sutun = x), hucre (x_px, y_px).
Dunya <-> piksel: hucre merkezi, harita orijini (sol alt kose) ve cozunurlukten.
"""
import numpy as np


class Costmap:
    def __init__(self, pixel_width, pixel_height, resolution, origin_x, origin_y):
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data = np.full((pixel_height, pixel_width), -1, np.int8)

    def update_data(self, data):
        self.data = data

    def pixelToPose(self, x_px, y_px):
        """Hucre (x_px, y_px) merkezi -> harita cercevesinde (x, y) [m]."""
        return (self.origin_x + (x_px + 0.5) * self.resolution,
                self.origin_y + (y_px + 0.5) * self.resolution)

    def poseToPixel(self, x, y):
        """Harita cercevesinde (x, y) [m] -> hucre (x_px, y_px)."""
        return (int(np.floor((x - self.origin_x) / self.resolution)),
                int(np.floor((y - self.origin_y) / self.resolution)))
