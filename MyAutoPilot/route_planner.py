from lane_change import LaneChange

class RoutePlanner():
    def __init__(self , y_far , y_near , lane_change_diff):
        self.control_priority = "center"
        self.lane_change_available = {"L": False , "R": False}
        self.lane_change = LaneChange(y_far , y_near , lane_change_diff)

    def update(self , center_points , cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points , lane_available , direction = None , width = 640):
        for side in ("L" , "R"):
            self.lane_change_available[side] = lane_available[side] and self.control_priority == "center"

        if direction in ("L" , "R") and self.lane_change_available[direction]:
            if self.lane_change.request(direction):
                self.control_priority = "lane_change"

        path_points = center_points
        if self.control_priority == "lane_change":
            if self.lane_change.direction == "L":
                current_lane_points = cl_lane_points
                neighbour_lane_points = nl_lane_points
            else:
                current_lane_points = cr_lane_points
                neighbour_lane_points = nr_lane_points

            path_points , should_exit = self.lane_change.update(current_lane_points , neighbour_lane_points , width)
            if should_exit:
                self.control_priority = "center"
                path_points = center_points

        for side in ("L" , "R"):
            self.lane_change_available[side] = lane_available[side] and self.control_priority == "center"

        return path_points
