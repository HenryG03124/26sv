import numpy as np

max_gap_far = 48 #px
max_gap_near = 48 #px
init_y_window = 80 #px
width = 640 #px
max_center_diff_bottom = 20 #px

class LaneRefiner():
    def __init__(self , y_far , y_near , lane_x_diff):
        self.lane_center = width // 2
        self.avalible_points = {"L" : 0 , "R" : 0}
        self.status = {"L" : "tracking" , "R" : "tracking"}
        self.y_far = y_far
        self.y_near = y_near
        self.lane_x_diff = lane_x_diff

    def calculate_slope(self , points):
        slope = 0
        if len(points) == 0 or len(points) == 1:
            return slope
        for i in range(len(points) - 1):
            slope += (points[i + 1][0] - points[i][0]) / (points[i + 1][1] - points[i][1])
        
        slope /= (len(points) - 1)
        return slope

    def initialize_sampling(self , points , sides , min_support_rows = 4 , init_x_diff = 4 , min_y_diff = 12):
        if len(points) < min_support_rows:
            self.status[sides] = "init failed: " + sides
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

                slope = self.calculate_slope([points[i] , points[j]])
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

    def sample_lane_points(self , mask , sgl_lane_px_offset , sample_interval):
        center_x = self.lane_center
        llane_sample_points = []
        rlane_sample_points = []
        llane_sample_points_init = []
        rlane_sample_points_init = []
        prev_llane_x_center = -2 * self.lane_x_diff
        prev_rlane_x_center = -2 * self.lane_x_diff
        prev_llane_y = 0
        prev_rlane_y = 0

        for y in range(self.y_near , self.y_far - 1 , -sample_interval):
            lane_x_indices = np.where(mask[y] == 3)[0]
            llane_slope = self.calculate_slope(llane_sample_points[-4 : ])
            rlane_slope = self.calculate_slope(rlane_sample_points[-4 : ])
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
                    llane_sample_points_init = [point for point in llane_sample_points_init if point[1] - y <= init_y_window][-8 : ] #create sliding windows (len = 8)
                    llane_sample_points = self.initialize_sampling(llane_sample_points_init , sides = "L")
                    if llane_sample_points:
                        prev_llane_x_center , prev_llane_y = llane_sample_points[-1]
                        llane_existance = True

                elif abs(llane_x_center - (prev_llane_x_center + llane_slope * (y - prev_llane_y))) <= self.lane_x_diff:
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
                    rlane_sample_points_init = [point for point in rlane_sample_points_init if point[1] - y <= init_y_window][-8 : ] #create sliding windows (len = 8)
                    rlane_sample_points = self.initialize_sampling(rlane_sample_points_init , sides = "R")
                    if rlane_sample_points:
                        prev_rlane_x_center , prev_rlane_y = rlane_sample_points[-1]
                        rlane_existance = True

                elif abs(rlane_x_center - (prev_rlane_x_center + rlane_slope * (y - prev_rlane_y))) <= self.lane_x_diff:
                    rlane_sample_points.append([rlane_x_center , y])
                    prev_rlane_x_center = rlane_x_center
                    prev_rlane_y = y
                    rlane_existance = True

            if llane_existance and rlane_existance:
                center_x = (llane_x_center + rlane_x_center) // 2
                if self.y_near - y <= max_center_diff_bottom:
                    self.lane_center = center_x

        return llane_sample_points , rlane_sample_points

    def fit_lane_points(self , sample_points , height , width , degree , sides):
        self.avalible_points[sides] = len(sample_points)
        if len(sample_points) < degree + 1:
            self.status[sides] = "not enough sample: " + sides
            return []

        x_values = np.array([point[0] for point in sample_points] , dtype = np.float64)
        y_values = np.array([point[1] for point in sample_points] , dtype = np.float64)
        weights = 1 - (y_values / height) ** 2

        if min(y_values) - self.y_far > max_gap_far or self.y_near - max(y_values) > max_gap_near:
            self.status[sides] = "not enough sample on ends: " + sides
            return []

        coefficients = np.polyfit(y_values , x_values , degree , w = weights)
        predicted_y_values = np.arange(self.y_far , self.y_near + 1)
        predicted_x_values = np.polyval(coefficients , predicted_y_values)
        predicted_x_values = np.rint(predicted_x_values).astype(np.int32)
        predicted_x_values = np.clip(predicted_x_values , 0 , width - 1)

        lane_points = [[int(x) , int(y)] for x , y in zip(predicted_x_values , predicted_y_values)]
        self.status[sides] = "tracking"
        return lane_points

    def calculate_center_points(self , llane_points , rlane_points):
        if not llane_points or not rlane_points:
            return []

        llane_yxdict = {y : x for x , y in llane_points}
        rlane_yxdict = {y : x for x , y in rlane_points}
        common_y_values = sorted(llane_yxdict.keys() & rlane_yxdict.keys() , reverse = True)
        center_points = [[(llane_yxdict[y] + rlane_yxdict[y]) // 2 , y] for y in common_y_values]

        return center_points

    def refine(self , mask , degree , sgl_lane_px_offset , return_sample = False):
        """Return left and right lane points, optionally preceded by both samples."""
        height , width = mask.shape
        llane_sample_points , rlane_sample_points = self.sample_lane_points(mask , sgl_lane_px_offset , 4)

        llane_points = self.fit_lane_points(llane_sample_points , height , width , degree , "L")
        rlane_points = self.fit_lane_points(rlane_sample_points , height , width , degree , "R")

        if return_sample:
            return llane_sample_points , rlane_sample_points , llane_points , rlane_points
        else:
            return llane_points , rlane_points
