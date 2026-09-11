import numpy as np
import cv2

class RoadCenter():
    def detect(self , mask):
        height , width = mask.shape
        center_points = []

        prev_center_x = width // 2
        for y in range(height // 2 , height - 10 , 5):
            lane_x = np.where(mask[y] == 3)[0]
            left = lane_x[lane_x < prev_center_x]
            right = lane_x[lane_x > prev_center_x]

            if len(left) == 0 or len(right) == 0:
                continue

            left_x = left[-1]
            right_x = right[0]
            lane_center_x = (left_x + right_x) // 2
            center_points.append((lane_center_x , y))
            prev_center_x = lane_center_x

        return center_points
