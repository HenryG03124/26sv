import numpy as np
import cv2

class RoadCenter():
    def detect_from_lr(mask , left_points , right_points):
        height , width = mask.shape
        if len(left_points) == 0 or len(right_points) == 0:
            return []

        left_yxdict = {y : x for x , y in left_points}
        right_yxdict = {y : x for x , y in right_points}
        common_y_values = sorted(left_yxdict.keys() & right_yxdict.keys() , reverse = True)

        center_points = [[width // 2 , height - 10]]
        for y in common_y_values:
            lane_center_x = (left_yxdict[y] + right_yxdict[y]) // 2
            center_points.append([lane_center_x , y])

        return center_points

    def detect_from_mask(mask , y_far , y_near):
        height , width = mask.shape
        center_points = [[width // 2 , height - 10]]

        prev_center_x = width // 2

        left_x = 0
        right_x = width - 1
        lane_width = 0
        unreliable_count = 0
        bottom = True

        for y in range(y_near , y_far - 1 , -4):
            road = np.where(mask[y] == 1)[0]
            lane = np.where(mask[y] == 3)[0]
            left = lane[lane < prev_center_x]
            right = lane[lane > prev_center_x]

            if len(road) == 0:
                unreliable_count += 1
                continue

            if len(left) > 0 and len(right) > 0:
                bottom = False
                left_x = left[-1]
                right_x = right[0]

                lane_center_x = (left_x + right_x) // 2
                lane_width = right_x - left_x
                unreliable_count = 0

            elif bottom:
                left_x = road[0]
                right_x = road[-1]
                
                lane_center_x = (left_x + right_x) // 2
                lane_width = right_x - left_x
                unreliable_count = 0

            elif lane_width > 0 and len(left) == 0 and len(right) > 0:
                right_x = right[0]
                lane_center_x = right_x - lane_width // 2
                unreliable_count += 1

            elif lane_width > 0 and len(right) == 0 and len(left) > 0:
                left_x = left[-1]
                lane_center_x = left_x + lane_width // 2
                unreliable_count += 1

            else:
                unreliable_count += 1
                continue

            if unreliable_count > 3:
                continue

            lane_center_x = int(np.clip(lane_center_x , 0 , width - 1))
            center_points.append([lane_center_x , y])
            prev_center_x = lane_center_x

        return center_points    

    def center_points_filter(center_points , center_offset_threshold):
        center_points = center_points[1 : len(center_points)]

        if len(center_points) < 2:
            return center_points

        prev = center_points[0]
        for i in range(len(center_points)):
            if center_points[i][0] - prev[0] > center_offset_threshold or prev[0] - center_points[i][0] > center_offset_threshold:
                center_points[i][0] = (prev[0] + center_points[i + 1][0]) // 2
            prev = center_points[i]

        return center_points
