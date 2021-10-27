#! /usr/bin/python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

#cell status
EMPTY = 0.01
UNKNOWN = 0.49
FILLED = 0.99

class Map:
    def __init__(self):
        self.msg = OccupancyGrid()
        self.msg.header.frame_id = 'base_link'
        self.msg.info.resolution = 0.1
        self.msg.info.width = int(30.0/0.1)
        self.msg.info.height = int(30.0/0.1)
        self.msg.info.origin.position.x = -10.0
        self.msg.info.origin.position.y = -10.0

        #default fill of map grid cells
        self.grid = self.log(UNKNOWN) * np.ones((int(30.0/0.1), int(30.0/0.1)))

        self.sub = rospy.Subscriber('/base_scan', LaserScan, self.callback, queue_size = 1)
        self.pub = rospy.Publisher('/map', OccupancyGrid, queue_size = 1)
        
    def callback(self, base_scan):
        angle_min = base_scan.angle_min
        angle_increment = base_scan.angle_increment

        for index, value in enumerate(base_scan.ranges):
            theta = angle_min + index * angle_increment
            x0 = -self.msg.info.origin.position.x / 0.1
            y0 = -self.msg.info.origin.position.y / 0.1
            x1 = (value * np.cos(theta) - self.msg.info.origin.position.x) / 0.1
            y1 = (value * np.sin(theta) - self.msg.info.origin.position.y) / 0.1
            d_cells = value / 0.1
            i, j = self.bresenham(y0, x0, y1, x1, d_cells)
            self.grid[int(i), int(j)] += self.log(FILLED) - self.log(UNKNOWN)

        generated_map = self.grid.flatten()
        self.msg.data = (self.probability(generated_map) * 100).astype(dtype = np.int8)
        self.pub.publish(self.msg)

    def probability(self, degree):
        return np.exp(degree) / (1.0 + np.exp(degree))

    def log(self, probability):
        return np.log(probability/(1 - probability))

    def is_point_inside (self, i, j):
        return i >= 0 and j >= 0 and i < self.grid.shape[0] and j < self.grid.shape[1]
        
    # Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    def bresenham(self, x0, y0, x1, y1, d):
        dx = abs(y1 - y0)
        sx = 1 if y0 < y1 else -1

        dy = -1 * abs(x1 - x0)
        sy = 1 if x0 < x1 else -1

        j, i = y0, x0
        err = dx + dy

        while True:
            if (j == y1 and i == x1) or (np.sqrt((j - y0) ** 2 + (i - x0) ** 2) >= d) or not self.is_point_inside(i, j):
                return i, j
            elif self.grid[int(i), int(j)] == 100:
                return i, j

            if self.is_point_inside(i, j):
                self.grid[int(i), int(j)] += self.log(EMPTY) - self.log(UNKNOWN)

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                j += sx
            if e2 <= dx:
                err += dx
                i += sy

rospy.init_node("BuildMap")
Map()
rospy.spin()
