import numpy as np
import cv2
import math

prev_weight = 0.8
raw_weight = 0.2
width = 640

class Steering():
    def existence_filter(center_points , y_far , y_near):
        existing_y = {y for x , y in center_points}
        return y_far in existing_y and y_near in existing_y

    def calculate_offset(center_points , y_far , y_near):
        xydict = {y : x for x , y in center_points}
        offset = (xydict[y_near] - width / 2) / (width / 2)
        return offset

    def calculate_hdg(center_points , y_far , y_near):
        xydict = {y : x for x , y in center_points}
        hdg = math.atan2(xydict[y_far] - xydict[y_near] , y_near - y_far)
        return hdg

    def steering(offset , hdg , K_offset , K_hdg , steering_prev):
        steering_raw = K_offset * offset + K_hdg * hdg
        steering_raw = np.clip(steering_raw , -1 , 1)

        steering = prev_weight * steering_prev + raw_weight * steering_raw

        return steering