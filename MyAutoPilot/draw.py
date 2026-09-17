import numpy as np
import cv2

class Draw():
    def draw_boxes(detector , frame , boxes , scores , od_colors):
        for box, score, label in zip(boxes , scores , detector.filtered_labels):
            x1, y1, x2, y2 = map(int , box)
            cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)
            cv2.putText(frame , text = detector.class_names[label] + f" {score:.2f}" , org = (x1 , max(y1 - 5 , 0)) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = 0.3 , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)

    def draw_road_center(frame , center_points , y_far , y_near):
        if len(center_points) < 3:
            return

        points = np.array(center_points , dtype = np.int32).reshape(-1 , 1 , 2)
        cv2.polylines(frame , [points] , isClosed = False , color = (255 , 0 , 0) , thickness = 1 , lineType = cv2.LINE_AA)

        yxdict = {y: x for x , y in center_points}

        for y in (y_far, y_near):
            if y in yxdict:
                cv2.circle(frame , center = (int(yxdict[y]) , int(y)) , radius = 5 , color = (0 , 255 , 255) , thickness = -1  ,lineType = cv2.LINE_AA)

        if y_near in yxdict:
            pts = np.array([[320 , 351] , [int(yxdict[y_near]) , int(y_near)]] , dtype = np.int32)
            cv2.polylines(frame , [pts] , isClosed = False , color = (255 , 0 , 0) , thickness = 1 , lineType = cv2.LINE_AA)

    def draw_lane_points(frame , llane_sample_points , rlane_sample_points):
        for point in llane_sample_points:
            cv2.circle(frame , center = (int(point[0]) , int(point[1])) , radius = 1 , color = (0 , 0 , 255) , thickness = -1  ,lineType = cv2.LINE_AA)
        for point in rlane_sample_points:
            cv2.circle(frame , center = (int(point[0]) , int(point[1])) , radius = 1 , color = (0 , 0 , 0) , thickness = -1  ,lineType = cv2.LINE_AA)

    def draw_lane_lines(frame , left_points , right_points):
        for lane_points in (left_points , right_points):
            if len(lane_points) < 3:
                continue

            points = np.array(lane_points , dtype = np.int32).reshape(-1 , 1 , 2)
            cv2.polylines(frame , [points] , isClosed = False , color = (0 , 255 , 255) , thickness = 1 , lineType = cv2.LINE_AA)
