import numpy as np

max_gap_far = 48 #px
max_gap_near = 48 #px
init_y_window = 80 #px
width = 640 #px
max_center_diff_bottom = 20 #px

class LaneRefiner():
    def __init__(self , current_y_far , current_y_near , lane_x_diff , neighbour_y_far , neighbour_y_near):
        self.lane_center = width // 2
        self.avalible_points = {"CL" : 0 , "CR" : 0 , "NL" : 0 , "NR" : 0}
        self.status = {"CL" : "tracking" , "CR" : "tracking" , "NL" : "tracking" , "NR" : "tracking" , "center" : "accurate"}
        self.current_y_far = current_y_far
        self.current_y_near = current_y_near
        self.lane_x_diff = lane_x_diff
        self.neighbour_y_far = neighbour_y_far
        self.neighbour_y_near = neighbour_y_near
        self.prev_lane_width = []

    def calculate_slope(self , points):
        slope = 0
        if len(points) == 0 or len(points) == 1:
            return slope
        for i in range(len(points) - 1):
            slope += (points[i + 1][0] - points[i][0]) / (points[i + 1][1] - points[i][1])
        
        slope /= (len(points) - 1)
        return slope

    def update_lane_width(self , cl_lane_points , cr_lane_points):
        cl_lane_yxdict = {y : x for x , y in cl_lane_points}
        cr_lane_yxdict = {y : x for x , y in cr_lane_points}
        common_y_values = sorted(cl_lane_yxdict.keys() & cr_lane_yxdict.keys() , reverse = True)
        self.prev_lane_width = [[cr_lane_yxdict[y] - cl_lane_yxdict[y] , y] for y in common_y_values]

    def sgl_lane_smapling(self , lane_x_indices , index , sides , sgl_lane_x_offset , min_neighbour_diff):
        if sides == "L":
            lane_x_indices = np.flip(lane_x_indices)

        prev = lane_x_indices[0]
        output = []
        temp = []
        for i in range(len(lane_x_indices)):
            if abs(lane_x_indices[i] - prev) > sgl_lane_x_offset:
                if len(output) == 0 or abs(temp[0] - output[-1][-1]) > min_neighbour_diff:
                    output.append(temp)
                temp = []

            temp.append(lane_x_indices[i])
            prev = lane_x_indices[i]

        if len(output) == 0 or abs(temp[0] - output[-1][-1]) > min_neighbour_diff:
            output.append(temp)
        if len(output) == 1:
            output.append([])

        return np.asarray(output[index] , dtype = np.float64)

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

    def sample_lane_points(self , mask , sgl_lane_x_offset , min_neighbour_diff , sample_interval):
        center_x = self.lane_center
        cl_lane_sample_points = []
        cr_lane_sample_points = []
        nl_lane_sample_points = []
        nr_lane_sample_points = []
        cl_lane_sample_points_init = []
        cr_lane_sample_points_init = []
        nl_lane_sample_points_init = []
        nr_lane_sample_points_init = []
        prev_cl_lane_x_center = -2 * self.lane_x_diff
        prev_cr_lane_x_center = -2 * self.lane_x_diff
        prev_nl_lane_x_center = -2 * self.lane_x_diff
        prev_nr_lane_x_center = -2 * self.lane_x_diff
        prev_cl_lane_y = 0
        prev_cr_lane_y = 0
        prev_nl_lane_y = 0
        prev_nr_lane_y = 0

        for y in range(self.current_y_near , self.current_y_far - 1 , -sample_interval):
            lane_x_indices = np.where(mask[y] == 3)[0]
            cl_lane_slope = self.calculate_slope(cl_lane_sample_points[-4 : ])
            cr_lane_slope = self.calculate_slope(cr_lane_sample_points[-4 : ])
            nl_lane_slope = self.calculate_slope(nl_lane_sample_points[-4 : ])
            nr_lane_slope = self.calculate_slope(nr_lane_sample_points[-4 : ])
            cl_lane_existance = False
            cr_lane_existance = False

            if len(lane_x_indices) == 0:
                continue

            left_x_indices = lane_x_indices[lane_x_indices < center_x]
            right_x_indices = lane_x_indices[lane_x_indices >= center_x]

            if len(left_x_indices) > 0:
                cl_lane_x_indices = self.sgl_lane_smapling(left_x_indices , 0 , "L" , sgl_lane_x_offset , min_neighbour_diff)
                cl_lane_x_center = np.mean(cl_lane_x_indices)
                if len(cl_lane_sample_points) == 0:
                    cl_lane_sample_points_init.append([cl_lane_x_center , y])
                    cl_lane_sample_points_init = [point for point in cl_lane_sample_points_init if point[1] - y <= init_y_window][-8 : ] #create sliding windows (len = 8)
                    cl_lane_sample_points = self.initialize_sampling(cl_lane_sample_points_init , sides = "CL")
                    if cl_lane_sample_points:
                        prev_cl_lane_x_center , prev_cl_lane_y = cl_lane_sample_points[-1]
                        cl_lane_existance = prev_cl_lane_y == y

                elif abs(cl_lane_x_center - (prev_cl_lane_x_center + cl_lane_slope * (y - prev_cl_lane_y))) <= self.lane_x_diff:
                    cl_lane_sample_points.append([cl_lane_x_center , y])
                    prev_cl_lane_x_center = cl_lane_x_center
                    prev_cl_lane_y = y
                    cl_lane_existance = True

            if len(right_x_indices) > 0:
                cr_lane_x_indices = self.sgl_lane_smapling(right_x_indices , 0 , "R" , sgl_lane_x_offset , min_neighbour_diff)
                cr_lane_x_center = np.mean(cr_lane_x_indices)
                if len(cr_lane_sample_points) == 0:
                    cr_lane_sample_points_init.append([cr_lane_x_center , y])
                    cr_lane_sample_points_init = [point for point in cr_lane_sample_points_init if point[1] - y <= init_y_window][-8 : ] #create sliding windows (len = 8)
                    cr_lane_sample_points = self.initialize_sampling(cr_lane_sample_points_init , sides = "CR")
                    if cr_lane_sample_points:
                        prev_cr_lane_x_center , prev_cr_lane_y = cr_lane_sample_points[-1]
                        cr_lane_existance = prev_cr_lane_y == y

                elif abs(cr_lane_x_center - (prev_cr_lane_x_center + cr_lane_slope * (y - prev_cr_lane_y))) <= self.lane_x_diff:
                    cr_lane_sample_points.append([cr_lane_x_center , y])
                    prev_cr_lane_x_center = cr_lane_x_center
                    prev_cr_lane_y = y
                    cr_lane_existance = True

            if self.neighbour_y_far <= y <= self.neighbour_y_near:
                if cl_lane_existance:
                    nl_lane_x_indices = self.sgl_lane_smapling(left_x_indices , 1 , "L" , sgl_lane_x_offset , min_neighbour_diff)
                    if len(nl_lane_x_indices) > 0:
                        nl_lane_x_center = np.mean(nl_lane_x_indices)
                        if len(nl_lane_sample_points) == 0:
                            nl_lane_sample_points_init.append([nl_lane_x_center , y])
                            nl_lane_sample_points_init = [point for point in nl_lane_sample_points_init if point[1] - y <= init_y_window][-8 : ]
                            nl_lane_sample_points = self.initialize_sampling(nl_lane_sample_points_init , sides = "NL")
                            if nl_lane_sample_points:
                                prev_nl_lane_x_center , prev_nl_lane_y = nl_lane_sample_points[-1]

                        elif abs(nl_lane_x_center - (prev_nl_lane_x_center + nl_lane_slope * (y - prev_nl_lane_y))) <= self.lane_x_diff:
                            nl_lane_sample_points.append([nl_lane_x_center , y])
                            prev_nl_lane_x_center = nl_lane_x_center
                            prev_nl_lane_y = y

                if cr_lane_existance:
                    nr_lane_x_indices = self.sgl_lane_smapling(right_x_indices , 1 , "R" , sgl_lane_x_offset , min_neighbour_diff)
                    if len(nr_lane_x_indices) > 0:
                        nr_lane_x_center = np.mean(nr_lane_x_indices)
                        if len(nr_lane_sample_points) == 0:
                            nr_lane_sample_points_init.append([nr_lane_x_center , y])
                            nr_lane_sample_points_init = [point for point in nr_lane_sample_points_init if point[1] - y <= init_y_window][-8 : ]
                            nr_lane_sample_points = self.initialize_sampling(nr_lane_sample_points_init , sides = "NR")
                            if nr_lane_sample_points:
                                prev_nr_lane_x_center , prev_nr_lane_y = nr_lane_sample_points[-1]

                        elif abs(nr_lane_x_center - (prev_nr_lane_x_center + nr_lane_slope * (y - prev_nr_lane_y))) <= self.lane_x_diff:
                            nr_lane_sample_points.append([nr_lane_x_center , y])
                            prev_nr_lane_x_center = nr_lane_x_center
                            prev_nr_lane_y = y

            if cl_lane_existance and cr_lane_existance:
                center_x = (cl_lane_x_center + cr_lane_x_center) // 2
                if self.current_y_near - y <= max_center_diff_bottom:
                    self.lane_center = center_x

        return cl_lane_sample_points , cr_lane_sample_points , nl_lane_sample_points , nr_lane_sample_points

    def fit_lane_points(self , sample_points , height , width , degree , sides):
        if sides in ("NL" , "NR"):
            y_far = self.neighbour_y_far
            y_near = self.neighbour_y_near
        else:
            y_far = self.current_y_far
            y_near = self.current_y_near

        self.avalible_points[sides] = len(sample_points)
        if len(sample_points) < degree + 1:
            self.status[sides] = "not enough sample: " + sides
            return []

        x_values = np.array([point[0] for point in sample_points] , dtype = np.float64)
        y_values = np.array([point[1] for point in sample_points] , dtype = np.float64)
        weights = 1 - (y_values / height) ** 2

        if min(y_values) - y_far > max_gap_far or y_near - max(y_values) > max_gap_near:
            self.status[sides] = "not enough sample on ends: " + sides
            return []

        coefficients = np.polyfit(y_values , x_values , degree , w = weights)
        predicted_y_values = np.arange(y_far , y_near + 1)
        predicted_x_values = np.polyval(coefficients , predicted_y_values)
        predicted_x_values = np.rint(predicted_x_values).astype(np.int32)
        predicted_x_values = np.clip(predicted_x_values , 0 , width - 1)

        lane_points = [[int(x) , int(y)] for x , y in zip(predicted_x_values , predicted_y_values)]
        self.status[sides] = "tracking"
        return lane_points

    def refine(self , mask , degree , sgl_lane_x_offset , min_neighbour_diff , return_sample = False):
        height , width = mask.shape
        cl_lane_sample_points , cr_lane_sample_points , nl_lane_sample_points , nr_lane_sample_points = self.sample_lane_points(mask , sgl_lane_x_offset , min_neighbour_diff , 4)

        cl_lane_points = self.fit_lane_points(cl_lane_sample_points , height , width , degree , "CL")
        cr_lane_points = self.fit_lane_points(cr_lane_sample_points , height , width , degree , "CR")
        nl_lane_points = self.fit_lane_points(nl_lane_sample_points , height , width , degree , "NL")
        nr_lane_points = self.fit_lane_points(nr_lane_sample_points , height , width , degree , "NR")

        if return_sample:
            return cl_lane_sample_points , cr_lane_sample_points , nl_lane_sample_points , nr_lane_sample_points , cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points
        else:
            return cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points

    def calculate_center_points(self , cl_lane_points , cr_lane_points):
        if not cl_lane_points and not cr_lane_points:
            self.status["center"] = "missing"
            return []
        elif not cl_lane_points:
            cr_lane_yxdict = {y : x for x , y in cr_lane_points}
            prev_lane_width_yxdict = {y : x for x , y in self.prev_lane_width}
            common_y_values = sorted(prev_lane_width_yxdict.keys() & cr_lane_yxdict.keys() , reverse = True)
            center_points = [[(2 * cr_lane_yxdict[y] - prev_lane_width_yxdict[y]) // 2 , y] for y in common_y_values]
            self.status["center"] = "perdicted" if center_points else "missing"
        elif not cr_lane_points:
            cl_lane_yxdict = {y : x for x , y in cl_lane_points}
            prev_lane_width_yxdict = {y : x for x , y in self.prev_lane_width}
            common_y_values = sorted(prev_lane_width_yxdict.keys() & cl_lane_yxdict.keys() , reverse = True)
            center_points = [[(2 * cl_lane_yxdict[y] + prev_lane_width_yxdict[y]) // 2 , y] for y in common_y_values]
            self.status["center"] = "predicted" if center_points else "missing"
        else:
            cl_lane_yxdict = {y : x for x , y in cl_lane_points}
            cr_lane_yxdict = {y : x for x , y in cr_lane_points}
            common_y_values = sorted(cl_lane_yxdict.keys() & cr_lane_yxdict.keys() , reverse = True)
            center_points = [[(cl_lane_yxdict[y] + cr_lane_yxdict[y]) // 2 , y] for y in common_y_values]
            self.update_lane_width(cl_lane_points , cr_lane_points)
            self.status["center"] = "accurate" if center_points else "missing"

        return center_points
