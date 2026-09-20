import numpy as np

history_time = 0.6 #seconds
max_history_gap = 0.5 #seconds, same expiry as MovDetector
min_history_time = 0.2 #seconds
confirm_time = 0.15 #seconds
release_time = 0.4 #seconds
lane_hold_time = 0.5 #seconds
overlap_threshold = 0.25
min_height_growth = 0.03
ttc_brake = 3.0 #seconds
ttc_full_brake = 1.2 #seconds
near_y_ratio = 0.84
near_height_ratio = 0.12
emergency_y_ratio = 0.90
emergency_height_ratio = 0.16

class AutoBrakes():
    def __init__(self):
        self.histories = {}
        self.risk_since = {}
        self.left_points = []
        self.right_points = []
        self.last_lane_time = -np.inf
        self.last_risk_time = -np.inf
        self.brake_value = 0.0
        self.target_id = None
        self.ttc = None
        self.lane_status = "missing"
        self.reason = "none"

    def update_lanes(self , left_points , right_points , timestamp):
        if len(left_points) >= 2 and len(right_points) >= 2:
            self.left_points = sorted(left_points , key = lambda point: point[1])
            self.right_points = sorted(right_points , key = lambda point: point[1])
            self.last_lane_time = timestamp
            self.lane_status = "current"
        elif timestamp - self.last_lane_time <= lane_hold_time:
            self.lane_status = "held"
        else:
            self.left_points = []
            self.right_points = []
            self.lane_status = "missing"

    def lane_x(self , points , y , width):
        points = np.asarray(points , dtype = np.float64)
        if y <= points[-1 , 1]:
            return float(np.interp(y , points[: , 1] , points[: , 0]))

        # Extend only the near end, using a short segment instead of the full polynomial.
        near_points = points[points[: , 1] >= points[-1 , 1] - 24]
        if len(near_points) < 2:
            near_points = points[-2 : ]
        slope = (near_points[-1 , 0] - near_points[0 , 0]) / (near_points[-1 , 1] - near_points[0 , 1])
        return float(np.clip(points[-1 , 0] + slope * (y - points[-1 , 1]) , 0 , width - 1))

    def in_lane(self , box , height , width):
        if not self.left_points or not self.right_points:
            return False

        x1 , y1 , x2 , y2 = box
        bottom_y = min(y2 , height - 1)
        far_y = max(self.left_points[0][1] , self.right_points[0][1])
        near_y = min(self.left_points[-1][1] , self.right_points[-1][1])
        if bottom_y < far_y or bottom_y > near_y + height * 0.2:
            return False

        left_x = self.lane_x(self.left_points , bottom_y , width)
        right_x = self.lane_x(self.right_points , bottom_y , width)
        if right_x <= left_x or x2 <= x1:
            return False

        overlap = max(0 , min(x2 , right_x) - max(x1 , left_x))
        return overlap / min(x2 - x1 , right_x - left_x) >= overlap_threshold

    def calculate_ttc(self , history): #time to crash
        if len(history) < 3 or history[-1][0] - history[0][0] < min_history_time:
            return None

        samples = np.asarray(history , dtype = np.float64)
        times = samples[: , 0] - samples[0 , 0]
        heights = samples[: , 1]
        height_rate = np.polyfit(times , heights , 1)[0]
        if height_rate <= 0 or height_rate * times[-1] < np.mean(heights) * min_height_growth:
            return None

        return float(heights[-1] / height_rate)

    def brake(self , left_points , right_points , trackings , timestamp , height , width):
        self.update_lanes(left_points , right_points , timestamp)
        active_ids = {track_id for track_id , tracking in trackings.items() if tracking["status"] == "tracked"}
        self.histories = {track_id: history for track_id , history in self.histories.items() if track_id in active_ids}
        self.risk_since = {track_id: start for track_id , start in self.risk_since.items() if track_id in active_ids}
        brake = 0.0
        self.target_id = None
        self.ttc = None
        self.reason = "none"

        for track_id , tracking in trackings.items():
            if tracking["status"] != "tracked":
                continue

            box = tracking["box"]
            box_height = box[3] - box[1]
            history = self.histories.get(track_id , [])
            if history and timestamp - history[-1][0] > max_history_gap:
                history = []
                self.risk_since.pop(track_id , None)
            history = [sample for sample in history if timestamp - sample[0] <= history_time]
            history.append((timestamp , box_height))
            self.histories[track_id] = history
            ttc = self.calculate_ttc(history)
            target_brake = 0.0
            target_reason = "none"
            emergency = False

            if self.in_lane(box , height , width):
                if ttc is not None and ttc < ttc_brake:
                    target_brake = float(np.clip((ttc_brake - ttc) / (ttc_brake - ttc_full_brake) , 0 , 1))
                    target_brake = 0.3 + 0.7 * target_brake
                    target_reason = "ttc"
                if box[3] >= height * near_y_ratio and box_height >= height * near_height_ratio:
                    if target_brake < 0.4:
                        target_reason = "near"
                    target_brake = max(target_brake , 0.4)
                emergency = box[3] >= height * emergency_y_ratio and box_height >= height * emergency_height_ratio
                if emergency:
                    target_brake = 1.0
                    target_reason = "emergency"

            if target_brake == 0:
                self.risk_since.pop(track_id , None)
                continue

            start = self.risk_since.setdefault(track_id , timestamp)
            if not emergency and timestamp - start < confirm_time:
                continue

            if target_brake > brake:
                brake = target_brake
                self.target_id = track_id
                self.ttc = ttc
                self.reason = target_reason

        if brake > 0:
            self.brake_value = brake
            self.last_risk_time = timestamp
        elif timestamp - self.last_risk_time > release_time:
            self.brake_value = 0.0
        elif self.brake_value > 0:
            self.reason = "hold"

        return self.brake_value
