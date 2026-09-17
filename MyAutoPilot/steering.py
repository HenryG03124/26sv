import numpy as np
import cv2
import math
import time

# Legacy exports; the active controller uses time-based rate limits instead.
prev_weight = 0.5
raw_weight = 0.5
width = 640
reference_y = 351 #px , same vehicle reference point as draw_road_center
L = 200
Ld = 12

# Low-speed image-space controller. K_steering = 3 uses these base gains.
offset_gain = 0.6
offset_rate_gain = 0.25
preview_gain = 0.08
offset_deadband = 2 / (width / 2)
offset_priority = 24 / (width / 2)
max_steering = 0.18
max_offset_rate_steering = 0.07
max_preview_steering = 0.015
steering_rate = 0.25 #joystick units / second
return_rate = 0.5 #joystick units / second
offset_rate_window = 0.35 #seconds
offset_rate_min_span = 0.15 #seconds
offset_rate_tau = 0.1 #seconds , the window already smooths the trend
reset_interval = 0.5 #seconds

class Steering():
    _prev_time = None
    _prev_offset = None
    _offset_rate = 0.0
    _offset_history = []

    def reset():
        Steering._prev_time = None
        Steering._reset_tracking()

    def _reset_tracking():
        Steering._prev_offset = None
        Steering._offset_rate = 0.0
        Steering._offset_history = []

    def _calculate_offset_rate(offset , current_time , elapsed):
        Steering._offset_history.append((current_time , offset))
        # Keep one point just before the window for low frame rates.
        cutoff_time = current_time - offset_rate_window
        while len(Steering._offset_history) > 2 and Steering._offset_history[1][0] <= cutoff_time:
            Steering._offset_history.pop(0)
        points = Steering._offset_history
        span = points[-1][0] - points[0][0]
        offset_rate = 0.0
        if len(points) >= 3 and span >= offset_rate_min_span:
            # Long-baseline pairs suppress frame-to-frame pixel jitter.
            slopes = []
            for i in range(len(points) - 1):
                for j in range(i + 1 , len(points)):
                    time_diff = points[j][0] - points[i][0]
                    if time_diff >= span / 2:
                        slopes.append((points[j][1] - points[i][1]) / time_diff)
            offset_rate = float(np.median(slopes))
            # Ignore roughly two pixels of net movement across the window.
            offset_rate = math.copysign(max(abs(offset_rate) - offset_deadband / span , 0) , offset_rate)
            offset_rate = np.clip(offset_rate , -1 , 1)
        rate_weight = 1 - math.exp(-elapsed / offset_rate_tau)
        Steering._offset_rate += rate_weight * (offset_rate - Steering._offset_rate)
        return Steering._offset_rate

    def _calculate_rate_steering(offset , offset_steering , gain):
        rate_steering = gain * offset_rate_gain * Steering._offset_rate
        priority_weight = min(abs(offset) / offset_priority , 1)
        priority_weight = priority_weight ** 2 * (3 - 2 * priority_weight)
        # Gradually restrict countersteering; no switch at the 24px boundary.
        correction_limit = min(max_offset_rate_steering , 0.7 * abs(offset_steering))
        rate_limit = (1 - priority_weight) * max_offset_rate_steering + priority_weight * correction_limit
        return float(np.clip(rate_steering , -rate_limit , rate_limit))

    def _validate_points(center_points):
        try:
            points = np.asarray(center_points , dtype = np.float64)
        except (TypeError , ValueError):
            return None
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            return None
        if not np.all(np.isfinite(points)):
            return None
        if np.any(points[: , 0] < 0) or np.any(points[: , 0] >= width):
            return None
        if np.any(points[: , 1] < 0) or np.any(points[: , 1] > reference_y):
            return None
        return points

    def existence_filter(center_points , y_far , y_near):
        points = Steering._validate_points(center_points)
        valid = points is not None and 0 <= y_far < y_near < reference_y
        if valid:
            existing_y = set(points[: , 1])
            valid = y_far in existing_y and y_near in existing_y
        if not valid:
            # main handles missing-lane output decay; discard the derivative history.
            Steering.reset()
        return valid

    def calculate_offset_near(center_points , y_near):
        yxdict = {y : x for x , y in center_points}
        offset_near = (yxdict[y_near] - width / 2) / (width / 2)
        return offset_near

    def calculate_offset_far(center_points , y_far):
        yxdict = {y : x for x , y in center_points}
        offset_far = (yxdict[y_far] - width / 2) / (width / 2)
        return offset_far

    def calculate_hdg(center_points , y_far , y_near):
        yxdict = {y : x for x , y in center_points}
        hdg = math.atan2(yxdict[y_far] - yxdict[y_near] , y_near - y_far)
        return hdg

    def calculate_target_angle(center_points , y_target):
        yxdict = {y : x for x , y in center_points}
        target_angle = math.atan2(yxdict[y_target] - width / 2 , reference_y - y_target)
        return target_angle

    def steering(center_points , y_far , K_steering , steering_prev):
        current_time = time.monotonic()
        elapsed = current_time - Steering._prev_time if Steering._prev_time is not None else 0.1
        if elapsed <= 0 or elapsed > reset_interval:
            Steering.reset()
            elapsed = 0.1
        dt = min(elapsed , 0.2)
        Steering._prev_time = current_time

        points = Steering._validate_points(center_points)
        if points is None or not np.isfinite(K_steering) or K_steering <= 0:
            # Discard path history without losing the output rate-limit clock.
            Steering._reset_tracking()
            return Steering._limit_steering(0 , steering_prev , dt)
        points = points[points[: , 1] >= y_far]
        if len(points) < 2 or y_far not in points[: , 1]:
            Steering._reset_tracking()
            return Steering._limit_steering(0 , steering_prev , dt)

        y_near = np.max(points[: , 1])
        if not 0 <= y_far < y_near < reference_y:
            Steering._reset_tracking()
            return Steering._limit_steering(0 , steering_prev , dt)

        # A short band avoids making the controller depend on one endpoint pixel.
        near_x = np.median(points[points[: , 1] >= y_near - 8 , 0])
        far_x = np.median(points[points[: , 1] <= y_far + 8 , 0])
        offset = (near_x - width / 2) / (width / 2)
        offset_far = (far_x - width / 2) / (width / 2)

        if Steering._prev_offset is not None:
            offset_diff = offset - Steering._prev_offset
            # A sudden path switch must not become a large derivative command.
            if abs(offset_diff) > 0.12 + 0.5 * dt:
                Steering._prev_offset = offset
                Steering._offset_rate = 0.0
                Steering._offset_history = [(current_time , offset)]
                return Steering._limit_steering(0 , steering_prev , dt)
        Steering._calculate_offset_rate(offset , current_time , elapsed)
        Steering._prev_offset = offset

        gain = min(K_steering / 3 , 2)
        offset_error = math.copysign(max(abs(offset) - offset_deadband , 0) , offset)
        offset_steering = gain * offset_gain * offset_error
        rate_steering = Steering._calculate_rate_steering(offset , offset_steering , gain)
        preview_weight = max(0 , 1 - abs(offset) / offset_priority)
        preview_steering = preview_weight * np.clip(gain * preview_gain * offset_far ,
                                                   -max_preview_steering , max_preview_steering)
        steering_raw = offset_steering + rate_steering + preview_steering
        return Steering._limit_steering(steering_raw , steering_prev , dt)

    def _limit_steering(steering_raw , steering_prev , dt):
        if not np.isfinite(steering_prev):
            steering_prev = 0.0
        steering_prev = float(np.clip(steering_prev , -max_steering , max_steering))
        steering_raw = float(np.clip(steering_raw , -max_steering , max_steering))
        # Return to neutral faster; build opposite lock only after reaching neutral.
        if steering_raw * steering_prev < 0:
            neutral_time = abs(steering_prev) / return_rate
            if dt <= neutral_time:
                return steering_prev - math.copysign(return_rate * dt , steering_prev)
            return math.copysign(min(abs(steering_raw) , steering_rate * (dt - neutral_time)) , steering_raw)
        rate = return_rate if abs(steering_raw) < abs(steering_prev) else steering_rate
        return float(steering_prev + np.clip(steering_raw - steering_prev , -rate * dt , rate * dt))

    def steering_pure_pursuit(center_points , y_far , y_near , steering_prev):
        # Compatibility entry point; the old uncalibrated pure-pursuit formula is retired.
        points = [[x , y] for x , y in center_points if y_far <= y <= y_near]
        return Steering.steering(points , y_far , 3 , steering_prev)
