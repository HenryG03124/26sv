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
from logger import Logger
from scs_telemetry import SCSTelemetry

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

y_far = 188 #px(240)
y_near = 292 #px(320)
K_steering = 3

monitor_index = 1
pixel_classifier_dataset = "bdd100k" #bdd100k / a2d2 / culane
road_center_mode = "lr"
calculate_steering = "enable"
calculate_brake = "enable"
control = "enable"

def main():
    steering = 0
    steering_prev = 0

    capturer = ScreenCapture(monitor_index)

    classifier = PixelClassifierV2()
    classifier.load_model(f"pc_model_v2_{pixel_classifier_dataset}.pth")

    detector = ObjectDetectorLite()
    detector.load_model("od_model_Lite.pth")
    mov_detector = MovDetector()

    road_center_detector = RoadCenter
    lane_refiner = LaneRefiner(y_far , y_near , lane_x_diff)
    auto_brakes = AutoBrakes()
    input_controller = InputController()
    logger = Logger()
    telemetry_reader = SCSTelemetry()

    previous_frame_start = None
    frame_interval = None

    try:
        while True:
            frame_start = time.perf_counter()
            if previous_frame_start is not None:
                interval = frame_start - previous_frame_start
                weight = 1 - np.exp(-interval)
                frame_interval = interval if frame_interval is None else frame_interval + weight * (interval - frame_interval)
            previous_frame_start = frame_start
            fps = None if frame_interval is None else 1 / frame_interval
            frame = capturer.capture_window("Euro Truck Simulator 2")
            timestamp = time.monotonic()

            telemetry = telemetry_reader.read()

            mask = classifier.predict(frame , lane_prob_threshold)

            pos , boxes , scores , labels = detector.predict(frame)
            boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , mask.shape[1] , mask.shape[0] , mask , car_mask_ratio_threshold)
            trackings = mov_detector.update(boxes , scores , labels , timestamp , spd)

            llane_sample_points , rlane_sample_points , left_points , right_points = lane_refiner.refine(mask , degree , sgl_lane_px_offset , return_sample = True)

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
                steering = Steering.steering(center_points , y_far , K_steering , steering_prev , speed_kmh = telemetry["speed_kmh"] , timestamp = timestamp , y_near = y_near)
                print(steering , end = " ")
                steering_prev = steering

            brake = 0.0
            if calculate_brake == "enable":
                brake = auto_brakes.brake(left_points , right_points , trackings , timestamp , mask.shape[0] , mask.shape[1])
                Draw.show_brakes(display_frame , brake , auto_brakes.target_id , auto_brakes.ttc , auto_brakes.lane_status)
                print(brake , end = "\n")
            else:
                print("\n")

            if control == "enable":
                if calculate_steering == "enable":
                    input_controller.steering_controller(steering)
                if calculate_brake == "enable":
                    input_controller.brake_controller(brake)

            control_timestamp = time.monotonic()

            logger.write(timestamp , {
                "image_size": [mask.shape[1] , mask.shape[0]] ,
                "pixel_classifier_dataset": pixel_classifier_dataset ,
                "road_center_mode": road_center_mode ,
                "y_far": y_far ,
                "y_near": y_near ,
                "lane_sample_points": {"L": llane_sample_points , "R": rlane_sample_points} ,
                "refiner_sample_count": lane_refiner.avalible_points ,
                "refiner_status": lane_refiner.status ,
                "refiner_y_range": {
                    side: [min(point[1] for point in points) , max(point[1] for point in points)] if points else None
                    for side , points in (("L" , llane_sample_points) , ("R" , rlane_sample_points))
                } ,
                "left_points": left_points ,
                "right_points": right_points ,
                "center_points": center_points ,
                "trackings": trackings ,
                "steering": steering ,
                "control_timestamp": control_timestamp ,
                "steering_debug": Steering.diagnostics if calculate_steering == "enable" else {} ,
                "telemetry": telemetry ,
                "fps": fps ,
                "brake": brake ,
                "brake_target_id": auto_brakes.target_id if calculate_brake == "enable" else None ,
                "ttc": auto_brakes.ttc if calculate_brake == "enable" else None ,
                "lane_status": auto_brakes.lane_status if calculate_brake == "enable" else "disabled" ,
                "brake_reason": auto_brakes.reason if calculate_brake == "enable" else "disabled" ,
                "calculate_steering": calculate_steering ,
                "calculate_brake": calculate_brake ,
                "control": control ,
                "steering_sent": control == "enable" and calculate_steering == "enable" ,
                "brake_sent": control == "enable" and calculate_brake == "enable" ,
            })

            Draw.show_refiner(display_frame , lane_refiner.status , lane_refiner.avalible_points)

            height , width = frame.shape[:2]
            display_frame = cv2.resize(display_frame , (width , height) , interpolation = cv2.INTER_LINEAR)

            Draw.show_fps(display_frame , fps)
            Draw.show_speed(display_frame , telemetry)

            cv2.imshow("Result" , display_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    finally:
        capturer.close()
        cv2.destroyAllWindows()
        input_controller.close()
        logger.close()
        telemetry_reader.close()

if __name__ == "__main__":
    main()
