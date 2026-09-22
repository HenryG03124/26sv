import cv2
import numpy as np
import time

from screen_capture import ScreenCapture
from draw import Draw
from pixel_classifier_v2 import PixelClassifierV2
from object_detector import ObjectDetectorLite
from mov_detector import MovDetector
from lane_refiner import LaneRefiner
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

sgl_lane_px_offset = 10 #px
degree = 2
lane_x_diff = 10 #px

spd = 70 #px / s

y_far = 220 #px(188)
y_near = 320 #px(292)
K_steering = 3

capture_fullscreen = False #True: full monitor , False: window client area
monitor_index = 1
window_name = "Euro Truck Simulator 2"
scale = 1

pixel_classifier_dataset = "bdd100k" #bdd100k / a2d2 / culane
control = "enable"

def main():
    steering = 0
    steering_prev = 0

    capturer = ScreenCapture(monitor_index)

    classifier = PixelClassifierV2()
    classifier.load_model(f"weights/pc_model_v2_{pixel_classifier_dataset}.pth")

    detector = ObjectDetectorLite()
    detector.load_model("weights/od_model_Lite.pth")
    mov_detector = MovDetector()

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
            frame = capturer.capture_frame() if capture_fullscreen else capturer.capture_window(window_name)
            timestamp = time.monotonic()

            telemetry = telemetry_reader.read()

            mask = classifier.predict(frame , lane_prob_threshold)

            pos , boxes , scores , labels = detector.predict(frame)
            boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , mask.shape[1] , mask.shape[0] , mask , car_mask_ratio_threshold)
            trackings = mov_detector.update(boxes , scores , labels , timestamp , spd)

            llane_sample_points , rlane_sample_points , llane_points , rlane_points = lane_refiner.refine(mask , degree , sgl_lane_px_offset , return_sample = True)
            center_points = lane_refiner.calculate_center_points(llane_points , rlane_points)

            current_lane_mask = np.zeros_like(mask , dtype = np.uint8)

            if len(llane_points) >= 3 and len(rlane_points) >= 3:
                polygon = np.array(llane_points + rlane_points[ : : -1] ,dtype = np.int32)
                cv2.fillPoly(current_lane_mask , [polygon] , color = 1)

            display_frame = cv2.resize(frame , (mask.shape[1] , mask.shape[0]) , interpolation = cv2.INTER_LINEAR)
            active = ((mask == 1) & (current_lane_mask > 0)) | (mask == 2) | (mask == 3)
            color_mask = pc_colors[mask]
            blended = cv2.addWeighted(display_frame , 1 - alpha , color_mask , alpha , 0)
            display_frame[active] = blended[active]

            Draw.draw_boxes(detector , display_frame , trackings , od_colors)

            Draw.draw_lane_points(display_frame , llane_sample_points , rlane_sample_points)
            Draw.draw_lane_lines(display_frame , llane_points , rlane_points)
            Draw.draw_road_center(display_frame , center_points , y_far , y_near)

            steering = Steering.steering(center_points , y_far , K_steering , steering_prev , speed_kmh = telemetry["speed_kmh"] , timestamp = timestamp , y_near = y_near)

            steering_prev = steering

            brake = 0.0
            brake = auto_brakes.brake(llane_points , rlane_points , trackings , timestamp , mask.shape[0] , mask.shape[1])
            print(steering , brake , end = "\n")

            if control == "enable":
                input_controller.steering_controller(steering)
                input_controller.brake_controller(brake)

            control_timestamp = time.monotonic()

            logger.write(timestamp , {
                "image_size": [mask.shape[1] , mask.shape[0]] ,
                "pixel_classifier_dataset": pixel_classifier_dataset ,
                "y_far": y_far ,
                "y_near": y_near ,
                "lane_sample_points": {"L": llane_sample_points , "R": rlane_sample_points} ,
                "refiner_sample_count": lane_refiner.avalible_points ,
                "refiner_status": lane_refiner.status ,
                "refiner_y_range": {
                    side: [min(point[1] for point in points) , max(point[1] for point in points)] if points else None
                    for side , points in (("L" , llane_sample_points) , ("R" , rlane_sample_points))
                } ,
                "llane_points": llane_points ,
                "rlane_points": rlane_points ,
                "center_points": center_points ,
                "trackings": trackings ,
                "steering": steering ,
                "control_timestamp": control_timestamp ,
                "steering_debug": Steering.diagnostics ,
                "telemetry": telemetry ,
                "fps": fps ,
                "brake": brake ,
                "brake_target_id": auto_brakes.target_id ,
                "ttc": auto_brakes.ttc ,
                "lane_status": auto_brakes.lane_status ,
                "brake_reason": auto_brakes.reason ,
                "control": control ,
                "steering_sent": control == "enable" ,
                "brake_sent": control == "enable" ,
            })

            height , width = frame.shape[:2]
            display_frame = cv2.resize(display_frame , (width // scale , height // scale) , interpolation = cv2.INTER_LINEAR)

            Draw.show_brakes(display_frame , brake , auto_brakes.target_id , auto_brakes.ttc , auto_brakes.lane_status)
            Draw.show_refiner(display_frame , lane_refiner.status , lane_refiner.avalible_points)
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
