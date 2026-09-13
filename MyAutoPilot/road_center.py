import numpy as np
import cv2

class RoadCenter():
    def detect(mask):
        height , width = mask.shape
        center_points = [[width // 2 , height - 10]]

        prev_center_x = width // 2

        left_x = 0
        right_x = width - 1
        for y in range(height - 15 , height // 2 - 10 , -5):
            road = np.where(mask[y] == 1)[0]
            lane = np.where(mask[y] == 3)[0]
            left = lane[lane < prev_center_x]
            right = lane[lane > prev_center_x]

            if len(road) == 0:
                continue

            if len(left) == 0 and len(right) == 0:
                left_x = road[-1]
                right_x = road[0]
            elif len(left) == 0 or len(right) == 0:
                pass
            else:
                left_x = left[-1]
                right_x = right[0]

            lane_center_x = (left_x + right_x) // 2
            center_points.append([lane_center_x , y])
            prev_center_x = lane_center_x

        return center_points

    def center_points_filter(center_points , center_offset_threshold):
        prev = center_points[0]
        for i in range(len(center_points) - 1):
            if center_points[i][0] - prev[0] > center_offset_threshold or prev[0] - center_points[i][0] > center_offset_threshold:
                center_points[i][0] = (prev[0] + center_points[i + 1][0]) // 2

        return center_points
