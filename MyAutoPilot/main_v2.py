import cv2
import numpy as np

from screen_capture import ScreenCapture
from pixel_classifier_v2 import PixelClassifierV2
from object_detector import ObjectDetector
from road_center_A import RoadCenterA
from road_center_B import RoadCenterB
from steering import Steering

pc_colors = np.array([[0 , 0 , 0] , [0 , 255 , 0] , [0 , 0 , 255] , [0 , 255 , 255]] , dtype = np.uint8)
od_colors = ((255 , 64 , 64) , (64 , 128 , 255))
class_names = ("pedestrian" , "car")

alpha = 0.3
car_mask_ratio_threshold = 0.6
lane_prob_threshold = 0.7
center_offset_threshold = 10 #px

y_far = 200 #px
y_near = 324 #px
K_offset = 1
K_hdg = 1

monitor_index = 1
obj_detect = "enable"
road_center = "enable"
road_center_algorithm = 1
steer = "enable"

def draw_boxes(detector , frame , mask , car_mask_ratio_threshold):
    pos , boxes , scores , labels = detector.predict(frame)
    boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , mask.shape[1] , mask.shape[0] , mask , car_mask_ratio_threshold)
    for box, score, label in zip(boxes , scores , labels):
        x1, y1, x2, y2 = map(int , box)
        cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)
        cv2.putText(frame , text = class_names[label] + f" {score:.2f}" , org = (x1 , max(y1 - 5, 0)) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = 0.3 , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)

def draw_road_center(frame , center_points , y_far , y_near):
    if len(center_points) < 3:
        return

    points = np.array(center_points , dtype = np.int32).reshape(-1 , 1 , 2)
    cv2.polylines(frame , [points] , isClosed = False , color = (255 , 0 , 0) , thickness = 1 , lineType = cv2.LINE_AA)

    xydict = {y: x for x, y in center_points}

    for y in (y_far, y_near):
        if y in xydict:
            cv2.circle(frame , center = (int(xydict[y]) , int(y)) , radius = 5 , color = (0 , 255 , 255) , thickness = -1  ,lineType = cv2.LINE_AA)

def main():
    steering = 0
    steering_prev = 0
    invalid_count = 0
    diff = 0

    capturer = ScreenCapture(monitor_index)

    classifier = PixelClassifierV2()
    classifier.load_model("pc_model_v2.pth")

    detector = ObjectDetector()
    detector.load_model("od_model_Lite.pth")

    if road_center_algorithm == 1:
        road_center_detector = RoadCenterA
    elif road_center_algorithm == 2:
        road_center_detector = RoadCenterB

    try:
        while True:
            frame = capturer.capture_frame()

            mask = classifier.predict(frame , lane_prob_threshold)

            display_frame = cv2.resize(frame , (mask.shape[1] , mask.shape[0]) , interpolation = cv2.INTER_LINEAR)
            active = mask != 0
            color_mask = pc_colors[mask]
            blended = cv2.addWeighted(display_frame , 1 - alpha , color_mask , alpha , 0)
            display_frame[active] = blended[active]

            if obj_detect == "enable":
                draw_boxes(detector , display_frame , mask , car_mask_ratio_threshold)

            if road_center == "enable":
                center_points = road_center_detector.detect(mask)
                center_points = road_center_detector.center_points_filter(center_points , center_offset_threshold)
                draw_road_center(display_frame , center_points , y_far , y_near)

            if steer == "enable" and road_center == "enable":
                if Steering.existence_filter(center_points, y_far, y_near):
                    invalid_count = 0
                    offset = Steering.calculate_offset(center_points , y_far , y_near)
                    hdg = Steering.calculate_hdg(center_points , y_far , y_near)
                    steering = Steering.steering(offset , hdg , K_offset , K_hdg , steering_prev)
                    diff = steering / 5 
                else:
                    if invalid_count < 5:
                        steering -= diff
                        invalid_count += 1
                    else:
                        steering = 0

                print(steering)
                steering_prev = steering

            width , height = capturer.native_resolution()
            display_frame = cv2.resize(display_frame , (width // 2 , height // 2) , interpolation = cv2.INTER_LINEAR)

            cv2.imshow("Result" , display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capturer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
