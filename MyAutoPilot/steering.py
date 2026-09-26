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

# Image-space preview controller. K_steering = 3 uses these base gains.
# Keep preview and output constraints separate, as in PythonRobotics path tracking.
# These are joystick/image units, not calibrated Stanley or pure-pursuit angles.
base_offset_gain = 0.45
offset_gain = 0.65
offset_rate_gain = 0.08
preview_gain = 0.08
offset_deadband = 2 / (width / 2)
correction_boost_start = 4 / (width / 2)
offset_priority = 24 / (width / 2)
max_steering = 0.45
max_offset_rate_steering = 0.02
max_preview_steering = 0.015
steering_rate = 0.40 #joystick units / second
return_rate = 0.5 #joystick units / second
offset_rate_window = 0.35 #seconds
offset_rate_min_span = 0.15 #seconds
offset_rate_tau = 0.1 #seconds , the window already smooths the trend
reset_interval = 0.5 #seconds
offset_tau = 0.1 #seconds
fallback_speed = 45 #km/h , conservative when telemetry is unavailable
lane_hold_time = 0.18 #seconds , bridge a brief detection dropout
lane_hold_distance = 3.0 #m , shorten the hold at high speed
path_confirm_time = 0.12 #seconds , reject isolated lane switches
path_confirm_tolerance = 0.06 #normalized image offset

class Steering():
    _prev_time = None
    _prev_offset = None
    _offset_rate = 0.0
    _offset_history = []
    _filtered_offset = None
    _last_valid_time = None
    _hold_until = None
    _pending_offset = None
    _pending_since = None
    diagnostics = {}

    def reset():
        Steering._prev_time = None
        Steering._last_valid_time = None
        Steering._hold_until = None
        Steering._reset_tracking()

    def _reset_tracking():
        Steering._prev_offset = None
        Steering._offset_rate = 0.0
        Steering._offset_history = []
        Steering._filtered_offset = None
        Steering._pending_offset = None
        Steering._pending_since = None
        Steering.diagnostics = {}

    def _unavailable(current_time , steering_prev , dt , reason = "missing" , clear_path = True):
        if clear_path:
            Steering._reset_tracking()
        # Never extend the deadline with another invalid frame, or build more lock.
        holding = Steering._hold_until is not None and current_time <= Steering._hold_until
        if holding:
            steering = float(np.clip(steering_prev , -max_steering , max_steering)) if np.isfinite(steering_prev) else 0.0
        else:
            # Only decay for the portion of this interval after the hold expired.
            decay_dt = dt if Steering._hold_until is None else min(dt , max(0 , current_time - Steering._hold_until))
            steering = Steering._limit_steering(0 , steering_prev , decay_dt)
        Steering.diagnostics = {
            "status": "held" if holding and reason == "missing" else reason ,
            "holding": holding , "steering_output": steering ,
            "path_age_s": None if Steering._last_valid_time is None else max(0 , current_time - Steering._last_valid_time) ,
        }
        return steering

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
        # Allow early countersteering, but never amplify pixel jitter into full lock.
        rate_limit = gain * max_offset_rate_steering
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
            # Compatibility validation helper; the active steering() manages dropout holds.
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

    def steering(center_points , y_far , K_steering , steering_prev , speed_kmh = None , timestamp = None , y_near = None):
        current_time = time.monotonic() if timestamp is None else timestamp
        if not np.isfinite(current_time):
            Steering.reset()
            Steering.diagnostics = {"status": "invalid_time" , "steering_output": 0.0}
            return 0.0
        elapsed = current_time - Steering._prev_time if Steering._prev_time is not None else 0.1
        if elapsed <= 0 or elapsed > reset_interval:
            Steering.reset()
            elapsed = 0.1
        dt = min(elapsed , 0.2)
        Steering._prev_time = current_time

        if not np.isfinite(K_steering) or K_steering <= 0:
            Steering._last_valid_time = None
            Steering._hold_until = None
            return Steering._unavailable(current_time , steering_prev , dt , "invalid_gain")
        points = Steering._validate_points(center_points)
        if points is None:
            return Steering._unavailable(current_time , steering_prev , dt)
        points = points[points[: , 1] >= y_far]
        if y_near is not None:
            points = points[points[: , 1] <= y_near]
            if y_near not in points[: , 1]:
                return Steering._unavailable(current_time , steering_prev , dt)
        if len(points) < 2 or y_far not in points[: , 1]:
            return Steering._unavailable(current_time , steering_prev , dt)

        y_near = np.max(points[: , 1])
        if not 0 <= y_far < y_near < reference_y:
            return Steering._unavailable(current_time , steering_prev , dt)

        # A short band avoids making the controller depend on one endpoint pixel.
        near_x = np.median(points[points[: , 1] >= y_near - 8 , 0])
        far_x = np.median(points[points[: , 1] <= y_far + 8 , 0])
        offset = (near_x - width / 2) / (width / 2)
        offset_far = (far_x - width / 2) / (width / 2)

        speed_fallback = speed_kmh is None or not np.isfinite(speed_kmh)
        speed = fallback_speed if speed_fallback else abs(speed_kmh)
        # The old quadratic attenuation removed most correction authority at 80+ km/h.
        speed_scale = 1 / math.hypot(1 , speed / 60)
        preview_weight = min(0.35 + speed / 150 , 0.7)
        # Both offsets use the same image units. This is NOT calibrated pure pursuit.
        offset_near = offset
        offset = (1 - preview_weight) * offset_near + preview_weight * offset_far

        path_reacquired = False
        if Steering._prev_offset is not None:
            offset_diff = offset - Steering._prev_offset
            # Keep the last accepted path as the reference until a new one persists.
            if abs(offset_diff) > 0.12 + 0.5 * dt:
                if Steering._pending_offset is None or abs(offset - Steering._pending_offset) > path_confirm_tolerance:
                    Steering._pending_offset = offset
                    Steering._pending_since = current_time
                Steering._offset_rate = 0.0
                Steering._offset_history = []
                if current_time - Steering._pending_since < path_confirm_time:
                    steering = Steering._unavailable(current_time , steering_prev , dt , "path_jump" , clear_path = False)
                    Steering.diagnostics["target_offset"] = float(offset)
                    return steering
                Steering._reset_tracking()
                path_reacquired = True
        Steering._pending_offset = None
        Steering._pending_since = None
        Steering._calculate_offset_rate(offset , current_time , elapsed)
        Steering._prev_offset = offset

        if Steering._filtered_offset is None:
            Steering._filtered_offset = offset
        weight = 1 - math.exp(-elapsed / offset_tau)
        Steering._filtered_offset += weight * (offset - Steering._filtered_offset)
        gain = min(K_steering / 3 , 2) * speed_scale
        offset_error = math.copysign(max(abs(Steering._filtered_offset) - offset_deadband , 0) , Steering._filtered_offset)
        # Preserve the old response to small errors, then smoothly add authority
        # between 4px and 24px. A tiny centerline wobble does not need the boost.
        correction_boost = float(np.clip((abs(Steering._filtered_offset) - correction_boost_start) /
                                         (offset_priority - correction_boost_start) , 0 , 1))
        effective_offset_gain = base_offset_gain * speed_scale + correction_boost * (offset_gain - base_offset_gain * speed_scale)
        offset_steering = gain * effective_offset_gain * offset_error
        # Increase sustained correction without increasing the old derivative/noise gain.
        rate_steering = Steering._calculate_rate_steering(offset , offset_steering , gain * speed_scale)
        steering_limit = max_steering / math.hypot(1 , speed / 70)
        steering_requested = offset_steering + rate_steering
        steering_raw = float(np.clip(steering_requested , -steering_limit , steering_limit))
        steering = Steering._limit_steering(steering_raw , steering_prev , dt)
        Steering._last_valid_time = current_time
        hold_duration = min(lane_hold_time , lane_hold_distance / max(speed / 3.6 , 0.1))
        Steering._hold_until = current_time + hold_duration
        Steering.diagnostics = {
            "status": "tracking" , "speed_kmh": float(speed) , "speed_fallback": speed_fallback ,
            "near_x": float(near_x) , "far_x": float(far_x) ,
            "preview_weight": float(preview_weight) , "target_offset": float(offset) ,
            "filtered_offset": float(Steering._filtered_offset) , "offset_rate": float(Steering._offset_rate) ,
            "offset_steering": float(offset_steering) , "rate_steering": float(rate_steering) ,
            "steering_raw": steering_raw , "steering_limit": float(steering_limit) ,
            "steering_requested": float(steering_requested) , "steering_output": steering ,
            "saturated": bool(abs(steering_requested) > steering_limit) ,
            "rate_limited": bool(abs(steering - steering_raw) > 1e-9) ,
            "path_reacquired": path_reacquired , "hold_duration_s": hold_duration ,
            "correction_boost": correction_boost ,
        }
        return steering

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
