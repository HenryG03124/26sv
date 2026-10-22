import numpy as np

class LaneChange():
    def __init__(self , y_far , y_near , diff):
        self.y_far = y_far
        self.y_near = y_near
        self.diff = diff
        self.direction = None
        self.exiting = False
        self.prev_current_lane = {}
        self.prev_neighbour_lane = {}

    def request(self , direction):
        if self.direction is not None or direction not in ("L" , "R"):
            return False

        self.direction = direction
        self.exiting = False
        self.prev_current_lane = {}
        self.prev_neighbour_lane = {}
        return True

    def update(self , current_lane_points , neighbour_lane_points , width):
        if self.direction is None:
            return [] , self.exiting

        current_lane = {y : x for x , y in current_lane_points}
        neighbour_lane = {y : x for x , y in neighbour_lane_points}
        if self.y_far not in current_lane or self.y_far not in neighbour_lane:
            self.exiting = True
            self.direction = None
            return [] , True

        current_x = current_lane[self.y_far]
        neighbour_x = neighbour_lane[self.y_far]
        valid = 0 <= current_x < width and 0 <= neighbour_x < width
        if self.direction == "L":
            valid = valid and neighbour_x < current_x
        else:
            valid = valid and neighbour_x > current_x
        if not valid:
            self.exiting = True
            self.direction = None
            return [] , True

        # Compare the same rows in consecutive frames.
        common_y_values = sorted(current_lane.keys() & neighbour_lane.keys() & self.prev_current_lane.keys() & self.prev_neighbour_lane.keys())
        if common_y_values:
            current_diff = np.median([current_lane[y] - self.prev_current_lane[y] for y in common_y_values])
            neighbour_diff = np.median([neighbour_lane[y] - self.prev_neighbour_lane[y] for y in common_y_values])
            if abs(current_diff) > self.diff and abs(neighbour_diff) > self.diff:
                self.exiting = True
                self.direction = None
                return [] , True

        self.prev_current_lane = current_lane
        self.prev_neighbour_lane = neighbour_lane

        near_x = width / 2
        far_x = (current_x + neighbour_x) / 2
        path_points = []
        for y in range(self.y_near , self.y_far - 1 , -1):
            weight = (self.y_near - y) / (self.y_near - self.y_far)
            x = near_x + weight * (far_x - near_x)
            path_points.append([x , y])

        return path_points , False
