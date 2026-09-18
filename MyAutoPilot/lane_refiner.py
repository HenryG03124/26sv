import numpy as np
import cv2

max_gap_far = 24 #px
max_gap_near = 24 #px

class LaneRefiner():
    def calculate_slope(points):
        slope = 0
        if len(points) == 0 or len(points) == 1:
            return slope
        for i in range(len(points) - 1):
            slope += (points[i + 1][0] - points[i][0]) / (points[i + 1][1] - points[i][1])
        
        slope /= (len(points) - 1)
        return slope

    def initialize_sampling(points , min_support_rows = 4 , init_x_diff = 4 , min_y_diff = 12):
        if len(points) < min_support_rows:
            return []

        points = np.asarray(points , dtype = np.float64)
        x_values = points[: , 0]
        y_values = points[: , 1]

        best_indices = []
        best_score = (-1 , -1 , -np.inf) #support rows , max(y) - min(y) , avg error

        for i in range(len(points) - 1):
            for j in range(i + 1 , len(points)):
                if abs(y_values[j] - y_values[i]) < min_y_diff:
                    continue

                slope = LaneRefiner.calculate_slope([points[i] , points[j]])
                pred_x = x_values[i] + slope * (y_values - y_values[i])
                errors = np.abs(x_values - pred_x)
                indices = np.flatnonzero(errors <= init_x_diff)

                score = (len(indices) , np.ptp(y_values[indices]) , -np.mean(errors[indices]))

                if score > best_score:
                    best_score = score
                    best_indices = indices

        if len(best_indices) < min_support_rows:
            return []

        return points[best_indices].tolist()

    def sample_lane_points(mask , y_far , y_near , sgl_lane_px_offset , sample_interval , lane_x_diff):
        width = mask.shape[1]
        center_x = width // 2
        llane_sample_points = []
        rlane_sample_points = []
        llane_sample_points_init = []
        rlane_sample_points_init = []
        prev_llane_x_center = -2 * lane_x_diff
        prev_rlane_x_center = -2 * lane_x_diff
        prev_llane_y = 0
        prev_rlane_y = 0

        for y in range(y_near , y_far - 1 , -sample_interval):
            lane_x_indices = np.where(mask[y] == 3)[0]
            llane_slope = LaneRefiner.calculate_slope(llane_sample_points[-4 : ])
            rlane_slope = LaneRefiner.calculate_slope(rlane_sample_points[-4 : ])
            llane_existance = False
            rlane_existance = False

            if len(lane_x_indices) == 0:
                continue

            llane_x_indices = lane_x_indices[lane_x_indices < center_x]
            rlane_x_indices = lane_x_indices[lane_x_indices >= center_x]

            if len(llane_x_indices) > 0:
                llane_x_edge = llane_x_indices[-1]
                current_llane_x_indices = llane_x_indices[llane_x_indices >= llane_x_edge - sgl_lane_px_offset]
                llane_x_center = np.mean(current_llane_x_indices)
                if len(llane_sample_points) == 0:
                    llane_sample_points_init.append([llane_x_center , y])
                    llane_sample_points_init = [point for point in llane_sample_points_init if point[1] - y <= 48][-8 : ] #create sliding windows (len = 8)
                    llane_sample_points = LaneRefiner.initialize_sampling(llane_sample_points_init)
                    if llane_sample_points:
                        prev_llane_x_center , prev_llane_y = llane_sample_points[-1]
                        llane_existance = True

                elif abs(llane_x_center - (prev_llane_x_center + llane_slope * (y - prev_llane_y))) <= lane_x_diff:
                    llane_sample_points.append([llane_x_center , y])
                    prev_llane_x_center = llane_x_center
                    prev_llane_y = y
                    llane_existance = True

            if len(rlane_x_indices) > 0:
                rlane_x_edge = rlane_x_indices[0]
                current_rlane_x_indices = rlane_x_indices[rlane_x_indices <= rlane_x_edge + sgl_lane_px_offset]
                rlane_x_center = np.mean(current_rlane_x_indices)
                if len(rlane_sample_points) == 0:
                    rlane_sample_points_init.append([rlane_x_center , y])
                    rlane_sample_points_init = [point for point in rlane_sample_points_init if point[1] - y <= 48][-8 : ] #create sliding windows (len = 8)
                    rlane_sample_points = LaneRefiner.initialize_sampling(rlane_sample_points_init)
                    if rlane_sample_points:
                        prev_rlane_x_center , prev_rlane_y = rlane_sample_points[-1]
                        rlane_existance = True

                elif abs(rlane_x_center - (prev_rlane_x_center + rlane_slope * (y - prev_rlane_y))) <= lane_x_diff:
                    rlane_sample_points.append([rlane_x_center , y])
                    prev_rlane_x_center = rlane_x_center
                    prev_rlane_y = y
                    rlane_existance = True

            if llane_existance and rlane_existance:
                center_x = (llane_x_center + rlane_x_center) // 2

        return llane_sample_points , rlane_sample_points

    def fit_lane_points(sample_points , height , width , y_far , y_near , degree):
        if len(sample_points) < degree + 1:
            return []

        x_values = np.array([point[0] for point in sample_points] , dtype = np.float64)
        y_values = np.array([point[1] for point in sample_points] , dtype = np.float64)
        weights = 1 - (y_values / height) ** 2

        if min(y_values) - y_far > max_gap_far or y_near - max(y_values) > max_gap_near:
            return []

        coefficients = np.polyfit(y_values , x_values , degree , w = weights)
        predicted_y_values = np.arange(y_far , y_near + 1)
        predicted_x_values = np.polyval(coefficients , predicted_y_values)
        predicted_x_values = np.rint(predicted_x_values).astype(np.int32)
        predicted_x_values = np.clip(predicted_x_values , 0 , width - 1)

        lane_points = [[int(x) , int(y)] for x , y in zip(predicted_x_values , predicted_y_values)]
        return lane_points

    def refine(mask , y_far , y_near , degree , sgl_lane_px_offset , lane_x_diff):
        height , width = mask.shape
        llane_sample_points , rlane_sample_points = LaneRefiner.sample_lane_points(mask , y_far , y_near , sgl_lane_px_offset , 4 , lane_x_diff)

        llane_points = LaneRefiner.fit_lane_points(llane_sample_points , height , width , y_far , y_near , degree)
        rlane_points = LaneRefiner.fit_lane_points(rlane_sample_points , height , width , y_far , y_near , degree)

        return llane_points , rlane_points
