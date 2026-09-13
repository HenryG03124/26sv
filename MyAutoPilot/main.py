import cv2
import numpy as np

from screen_capture import ScreenCapture
from pixel_classifier import PixelClassifier
from object_detector import ObjectDetector
from road_center import RoadCenter

pc_colors = np.array([[0 , 0 , 0] , [0 , 255 , 0] , [0 , 0 , 255]] , dtype = np.uint8)
od_colors = ((255 , 64 , 64) , (64 , 128 , 255))
class_names = ("pedestrian" , "car")
alpha = 0.3
car_mask_ratio_threshold = 0.6
monitor_index = 2
obj_detect = "enable"

def draw_boxes(detector , frame , mask , car_mask_ratio_threshold):
    pos , boxes , scores , labels = detector.predict(frame)
    boxes , scores , labels = detector.box_filter(pos , boxes , scores , labels , 448 , 256 , mask , car_mask_ratio_threshold)
    for box, score, label in zip(boxes , scores , labels):
        x1, y1, x2, y2 = map(int , box)
        cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)
        cv2.putText(frame , text = class_names[label] + f" {score:.2f}" , org = (x1 , max(y1 - 5, 0)) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = 0.3 , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)

def main():
    capturer = ScreenCapture(monitor_index)

    classifier = PixelClassifier()
    classifier.load_model("pc_model_U-Net.pth")

    detector = ObjectDetector()
    detector.load_model("od_model_Lite.pth")

    try:
        while True:
            frame = capturer.capture_frame()

            mask = classifier.predict(frame)

            display_frame = cv2.resize(frame , (mask.shape[1] , mask.shape[0]) , interpolation = cv2.INTER_LINEAR)
            active = mask != 0
            color_mask = pc_colors[mask]
            blended = cv2.addWeighted(display_frame , 1 - alpha , color_mask , alpha , 0)
            display_frame[active] = blended[active]

            if obj_detect == "enable":
                draw_boxes(detector , display_frame , mask , car_mask_ratio_threshold)

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
