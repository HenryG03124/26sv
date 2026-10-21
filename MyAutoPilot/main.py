import cv2
import numpy as np
import time
import ctypes

from screen_capture import ScreenCapture
from draw import Draw
from pixel_classifier_v2 import PixelClassifierV2
from object_detector import ObjectDetectorLite
from mov_detector import MovDetector
from lane_refiner import LaneRefiner
from route_planner import RoutePlanner
from steering import Steering
from auto_brakes import AutoBrakes
from throttle import Throttle
from input_controller import InputController
from auto_config import auto_config
from logger import Logger
from scs_telemetry import SCSTelemetry

# Display colors
pc_colors = np.array([[0 , 0 , 0] , [0 , 255 , 0] , [0 , 0 , 255] , [0 , 255 , 255]] , dtype = np.uint8)
od_colors = ((255 , 64 , 64) , (64 , 128 , 255) , (64 , 192 , 255))

# Segmentation thresholds
alpha = 0.3
car_mask_ratio_threshold = 0.5
lane_prob_threshold = 0.7

# Lane sampling and lane change
sgl_lane_x_offset = 10 #px
degree = 2
lane_x_diff = 10 #px
min_neighbour_diff = 30 #px
lane_change_diff = 30 #px , frame-to-frame lane shift

# Motion detection
spd = 70 #px / s

# Path sampling and steering
current_y_far = 200 #px(188)
current_y_near = 320 #px(292)
neighbour_y_far = 188 #px
neighbour_y_near = 248 #px
K_steering = 3

# Screen capture
capture_fullscreen = False #True: full monitor , False: window client area
monitor_index = 2
window_name = "Euro Truck Simulator 2"
scale = 1

# Control output
control = "enable"

# ETS2 controls
autoconfig = "disable"
controls_path = r"C:\Users\HenryGuo\OneDrive\文档\Euro Truck Simulator 2\steam_profiles\6A68656E727967756F\controls.sii"

def main():
    # ETS2 controls
    if autoconfig == "enable":
        auto_config(controls_path)

    # Runtime state
    steering = 0
    steering_prev = 0
    center_tracking_frames = 0
    left_pressed_prev = False
    right_pressed_prev = False

    # Keyboard input
    user32 = ctypes.WinDLL("user32" , use_last_error = True)
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short

    # Screen capture
    capturer = ScreenCapture(monitor_index)

    # Lane segmentation
    classifier = PixelClassifierV2()
    classifier.load_model("weights/pc_model_v2_bdd100k.pth")

    # Object detection and tracking
    detector = ObjectDetectorLite()
    detector.load_model("weights/od_model_Lite.pth")
    mov_detector = MovDetector()

    # Planning and control
    lane_refiner = LaneRefiner(current_y_far , current_y_near , lane_x_diff , neighbour_y_far , neighbour_y_near)
    route_planner = RoutePlanner(current_y_far , current_y_near , lane_change_diff)
    auto_brakes = AutoBrakes()
    throttle_controller = Throttle()
    input_controller = InputController()

    # Logging and telemetry
    logger = Logger()
    telemetry_reader = SCSTelemetry()

    # Frame timing
    previous_frame_start = None
    frame_interval = None

    try:
        while True:
            # Frame timing
            frame_start = time.perf_counter()
            if previous_frame_start is not None:
                interval = frame_start - previous_frame_start
                weight = 1 - np.exp(-interval)
                frame_interval = interval if frame_interval is None else frame_interval + weight * (interval - frame_interval)
            previous_frame_start = frame_start
            fps = None if frame_interval is None else 1 / frame_interval

            # Frame capture and telemetry
            frame = capturer.capture_frame() if capture_fullscreen else capturer.capture_window(window_name)
            timestamp = time.monotonic()

            telemetry = telemetry_reader.read()

            # Road and lane segmentation
            mask = classifier.predict(frame , lane_prob_threshold)

            # Object detection and tracking
            pos , boxes , scores , labels = detector.predict(frame)
            boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , mask.shape[1] , mask.shape[0] , mask , car_mask_ratio_threshold)
            trackings = mov_detector.update(boxes , scores , labels , timestamp , spd)

            # Lane change keys
            left_pressed = bool(user32.GetAsyncKeyState(0x25) & 0x8000)
            right_pressed = bool(user32.GetAsyncKeyState(0x27) & 0x8000)
            direction = None
            if left_pressed and not left_pressed_prev and not right_pressed:
                direction = "L"
            elif right_pressed and not right_pressed_prev and not left_pressed:
                direction = "R"
            left_pressed_prev = left_pressed
            right_pressed_prev = right_pressed

            # Lane refinement
            if route_planner.control_priority == "lane_change" or direction is not None:
                lane_refiner.lane_center = mask.shape[1] // 2
            cl_lane_sample_points , cr_lane_sample_points , nl_lane_sample_points , nr_lane_sample_points , cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points = lane_refiner.refine(mask , degree , sgl_lane_x_offset , min_neighbour_diff , return_sample = True)
            center_points = lane_refiner.calculate_center_points(cl_lane_points , cr_lane_points)

            # Route planning
            path_points = route_planner.update(center_points , cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points , lane_refiner.lane_available , direction = direction , width = mask.shape[1])
            if path_points:
                center_tracking_frames += 1
            else:
                center_tracking_frames = 0

            # Current lane mask
            current_lane_mask = np.zeros_like(mask , dtype = np.uint8)

            if len(cl_lane_points) >= 3 and len(cr_lane_points) >= 3:
                polygon = np.array(cl_lane_points + cr_lane_points[ : : -1] , dtype = np.int32)
                cv2.fillPoly(current_lane_mask , [polygon] , color = 1)

            # Segmentation overlay
            display_frame = cv2.resize(frame , (mask.shape[1] , mask.shape[0]) , interpolation = cv2.INTER_LINEAR)
            active = ((mask == 1) & (current_lane_mask > 0)) | (mask == 2) | (mask == 3)
            color_mask = pc_colors[mask]
            blended = cv2.addWeighted(display_frame , 1 - alpha , color_mask , alpha , 0)
            display_frame[active] = blended[active]

            # Lane and path drawing
            Draw.draw_lane_points(display_frame , cl_lane_sample_points , cr_lane_sample_points , nl_lane_sample_points , nr_lane_sample_points)
            Draw.draw_lane_lines(display_frame , cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points)
            Draw.draw_road_center(display_frame , path_points , current_y_far , current_y_near)

            # Steering
            steering = Steering.steering(path_points , current_y_far , K_steering , steering_prev , speed_kmh = telemetry["speed_kmh"] , timestamp = timestamp , y_near = current_y_near)

            steering_prev = steering

            # Braking and throttle
            brake = 0.0
            brake = auto_brakes.brake(cl_lane_points , cr_lane_points , trackings , timestamp , mask.shape[0] , mask.shape[1] , lane_refiner.lane_available["current"])
            throttle = throttle_controller.throttle(telemetry , center_tracking_frames , brake , timestamp)
            print("steering:" , steering , "brake:" , brake , "throttle:" , throttle , flush = True)

            # Control output
            if control == "enable":
                input_controller.steering_controller(steering)
                input_controller.throttle_controller(throttle)
                input_controller.brake_controller(brake)

            control_timestamp = time.monotonic()

            # Frame logging
            logger.write(timestamp , {
                "image_size": [mask.shape[1] , mask.shape[0]] ,
                "pixel_classifier_dataset": "bdd100k" ,
                "y_far": current_y_far ,
                "y_near": current_y_near ,
                "neighbour_y_far": neighbour_y_far ,
                "neighbour_y_near": neighbour_y_near ,
                "min_neighbour_diff": min_neighbour_diff ,
                "lane_sample_points": {"CL": cl_lane_sample_points , "CR": cr_lane_sample_points , "NL": nl_lane_sample_points , "NR": nr_lane_sample_points} ,
                "refiner_sample_count": lane_refiner.avalible_points ,
                "refiner_status": lane_refiner.status ,
                "lane_available": lane_refiner.lane_available ,
                "lane_tracking_frames": lane_refiner.tracking_frames ,
                "center_tracking_frames": center_tracking_frames ,
                "refiner_y_range": {
                    side: [min(point[1] for point in points) , max(point[1] for point in points)] if points else None
                    for side , points in (("CL" , cl_lane_sample_points) , ("CR" , cr_lane_sample_points) , ("NL" , nl_lane_sample_points) , ("NR" , nr_lane_sample_points))
                } ,
                "CL_lane_points": cl_lane_points ,
                "CR_lane_points": cr_lane_points ,
                "NL_lane_points": nl_lane_points ,
                "NR_lane_points": nr_lane_points ,
                "center_points": center_points ,
                "path_points": path_points ,
                "control_priority": route_planner.control_priority ,
                "lane_change_available": route_planner.lane_change_available ,
                "lane_change_direction": route_planner.lane_change.direction ,
                "lane_change_should_exit": route_planner.lane_change.should_exit ,
                "lane_change_diff": lane_change_diff ,
                "trackings": trackings ,
                "steering": steering ,
                "control_timestamp": control_timestamp ,
                "steering_debug": Steering.diagnostics ,
                "telemetry": telemetry ,
                "fps": fps ,
                "brake": brake ,
                "throttle": throttle ,
                "brake_target_id": auto_brakes.target_id ,
                "ttc": auto_brakes.ttc ,
                "lane_status": auto_brakes.lane_status ,
                "brake_reason": auto_brakes.reason ,
                "control": control ,
                "steering_sent": control == "enable" ,
                "brake_sent": control == "enable" ,
                "throttle_sent": control == "enable" ,
            })

            # Display scaling
            height , width = frame.shape[:2]
            display_frame = cv2.resize(display_frame , (width // scale , height // scale) , interpolation = cv2.INTER_LINEAR)

            # Object overlay
            scale_x = display_frame.shape[1] / mask.shape[1]
            scale_y = display_frame.shape[0] / mask.shape[0]
            Draw.draw_boxes(ObjectDetectorLite , display_frame , trackings , od_colors , scale_x , scale_y)

            # Status display
            Draw.show_brakes(display_frame , brake , auto_brakes.target_id , auto_brakes.ttc , auto_brakes.lane_status)
            Draw.show_refiner(display_frame , lane_refiner.status , lane_refiner.avalible_points)
            Draw.show_priority(display_frame , route_planner.control_priority)
            Draw.show_available(display_frame , route_planner.lane_change_available)
            Draw.show_fps(display_frame , fps)
            Draw.show_speed(display_frame , telemetry)
            Draw.show_throttle(display_frame , throttle)

            # Window and exit key
            cv2.imshow("Result" , display_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        capturer.close()
        cv2.destroyAllWindows()
        input_controller.close()
        logger.close()
        telemetry_reader.close()

if __name__ == "__main__":
    main()
