#Algorithm B: scan from anchor to 2 ends(y = 332 and y = 160)
import numpy as np
import cv2

anchor_y = 280 #px

class RoadCenterB():
    def scan_direction(mask , y_values , prev_center_x , lane_width):
        width = mask.shape[1]
        center_points = []
        unreliable_count = 0

        for y in y_values:
            road = np.where(mask[y] == 1)[0]
            lane = np.where(mask[y] == 3)[0]
            left = lane[lane < prev_center_x]
            right = lane[lane > prev_center_x]

            if len(road) == 0:
                unreliable_count += 1
                continue

            if len(left) > 0 and len(right) > 0:
                left_x = left[-1]
                right_x = right[0]

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

    def detect(mask):
        height , width = mask.shape
        scan_rows = list(range(height - 20 , height // 2 - 20 , -4))

        if len(scan_rows) == 0:
            return []

        anchor = min(scan_rows , key = lambda y : abs(y - anchor_y))

        road = np.where(mask[anchor] == 1)[0]
        lane = np.where(mask[anchor] == 3)[0]
        left = lane[lane < width // 2]
        right = lane[lane > width // 2]

        if len(road) == 0 or len(left) == 0 or len(right) == 0:
            return []

        left_x = left[-1]
        right_x = right[0]
        lane_center_x = (left_x + right_x) // 2
        lane_width = right_x - left_x

        center_points = [[lane_center_x , anchor]]

        near_rows = range(anchor + 4 , scan_rows[0] + 1 , 4)
        far_rows = range(anchor - 4 , scan_rows[-1] - 1 , -4)

        near_points = RoadCenterB.scan_direction(mask , near_rows , lane_center_x , lane_width)
        far_points = RoadCenterB.scan_direction(mask , far_rows , lane_center_x , lane_width)

        center_points.extend(near_points)
        center_points.extend(far_points)
        center_points.sort(key = lambda point: point[1] , reverse = True)

        return center_points

    def center_points_filter(center_points , center_offset_threshold):
        if len(center_points) < 2:
            return center_points

        prev = center_points[0]
        for i in range(1 , len(center_points) - 1):
            if center_points[i][0] - prev[0] > center_offset_threshold or prev[0] - center_points[i][0] > center_offset_threshold:
                center_points[i][0] = (prev[0] + center_points[i + 1][0]) // 2
            prev = center_points[i]

        return center_points #y = 332 , 328 , ...... , 164 , 160
