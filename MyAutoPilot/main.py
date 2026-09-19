import cv2
import numpy as np
import time

from screen_capture import ScreenCapture
from draw import Draw
from pixel_classifier_v2 import PixelClassifierV2
from object_detector import ObjectDetectorLite
from mov_detector import MovDetector
from lane_refiner import LaneRefiner
from road_center import RoadCenter
from steering import Steering
from auto_brakes import AutoBrakes
from input_controller import InputController

pc_colors = np.array([[0 , 0 , 0] , [0 , 255 , 0] , [0 , 0 , 255] , [0 , 255 , 255]] , dtype = np.uint8)
od_colors = ((255 , 64 , 64) , (64 , 128 , 255))

alpha = 0.3
car_mask_ratio_threshold = 0.5
lane_prob_threshold = 0.7

center_offset_threshold = 10 #px
sgl_lane_px_offset = 10 #px
degree = 2
lane_x_diff = 10 #px

spd = 70 #px / s

y_far = 188 #px
y_near = 292 #px
K_steering = 3

monitor_index = 1
road_center_mode = "lr"
calculate_steering = "enable"
calculate_brake = "enable"
control = "enable"

def main():
    steering = 0
    steering_prev = 0
    invalid_count = 0
    steering_diff = 0

    capturer = ScreenCapture(monitor_index)

    classifier = PixelClassifierV2()
    classifier.load_model("pc_model_v2.pth")

    detector = ObjectDetectorLite()
    detector.load_model("od_model_Lite.pth")
    mov_detector = MovDetector()

    road_center_detector = RoadCenter

    auto_brakes = AutoBrakes()

    input_controller = InputController()

    try:
        while True:
            frame = capturer.capture_frame()
            timestamp = time.monotonic()

            mask = classifier.predict(frame , lane_prob_threshold)

            pos , boxes , scores , labels = detector.predict(frame)
            boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , mask.shape[1] , mask.shape[0] , mask , car_mask_ratio_threshold)
            trackings = mov_detector.update(boxes , scores , labels , timestamp , spd)

            llane_sample_points , rlane_sample_points = LaneRefiner.sample_lane_points(mask , y_far , y_near , sgl_lane_px_offset , 4 , lane_x_diff)
            left_points , right_points = LaneRefiner.refine(mask , y_far , y_near , degree , sgl_lane_px_offset , lane_x_diff)

            current_lane_mask = np.zeros_like(mask , dtype = np.uint8)

            if len(left_points) >= 3 and len(right_points) >= 3:
                polygon = np.array(left_points + right_points[ : : -1] ,dtype = np.int32)
                cv2.fillPoly(current_lane_mask , [polygon] , color = 1)

            display_frame = cv2.resize(frame , (mask.shape[1] , mask.shape[0]) , interpolation = cv2.INTER_LINEAR)
            active = ((mask == 1) & (current_lane_mask > 0)) | (mask == 2) | (mask == 3)
            color_mask = pc_colors[mask]
            blended = cv2.addWeighted(display_frame , 1 - alpha , color_mask , alpha , 0)
            display_frame[active] = blended[active]

            Draw.draw_boxes(detector , display_frame , trackings , od_colors)

            if road_center_mode == "lr":
                Draw.draw_lane_points(display_frame , llane_sample_points , rlane_sample_points)

                Draw.draw_lane_lines(display_frame , left_points , right_points)

                center_points = road_center_detector.detect_from_lr(mask , left_points , right_points)
                center_points = road_center_detector.center_points_filter(center_points , center_offset_threshold)
                Draw.draw_road_center(display_frame , center_points , y_far , y_near)

            elif road_center_mode == "mask":
                center_points = road_center_detector.detect_from_mask(mask , y_far , y_near)
                center_points = road_center_detector.center_points_filter(center_points , center_offset_threshold)
                Draw.draw_road_center(display_frame , center_points , y_far , y_near)

            if calculate_steering == "enable":
                if Steering.existence_filter(center_points, y_far, y_near):
                    invalid_count = 0
                    steering = Steering.steering(center_points , y_far , K_steering , steering_prev)
                    steering_diff = steering / 5 
                else:
                    if invalid_count < 5:
                        steering -= steering_diff
                        invalid_count += 1
                    else:
                        steering = 0

                print(steering)
                steering_prev = steering

            brake = 0.0
            if calculate_brake == "enable":
                brake = auto_brakes.brake(left_points , right_points , trackings , timestamp , mask.shape[0] , mask.shape[1])
                Draw.draw_brake(display_frame , brake , auto_brakes.target_id , auto_brakes.ttc , auto_brakes.lane_status)

            if control == "enable":
                if calculate_steering == "enable":
                    input_controller.steering_controller(steering)
                input_controller.brake_controller(brake)

            width , height = capturer.native_resolution()
            display_frame = cv2.resize(display_frame , (width // 2 , height // 2) , interpolation = cv2.INTER_LINEAR)

            cv2.imshow("Result" , display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capturer.close()
        cv2.destroyAllWindows()
        input_controller.close()

if __name__ == "__main__":
    main()
