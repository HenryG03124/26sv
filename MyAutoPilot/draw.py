import numpy as np
import cv2

fontScale = 0.8
text_margin = 12
text_top = 30
text_line_height = 32

class Draw():
    def show_fps(frame , fps):
        text = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
        text_width = cv2.getTextSize(text , cv2.FONT_HERSHEY_SIMPLEX , fontScale , 1)[0][0]
        position = (frame.shape[1] - text_width - text_margin , text_top)
        cv2.putText(frame , text = text , org = position , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = fontScale , color = (0 , 255 , 255) , thickness = 1 , lineType = cv2.LINE_AA)

    def show_speed(frame , telemetry):
        speed = telemetry["speed_kmh"]
        text = f"SPEED: {speed:.1f} km/h" if speed is not None else f"SPEED: -- ({telemetry['status']})"
        text_width = cv2.getTextSize(text , cv2.FONT_HERSHEY_SIMPLEX , fontScale , 1)[0][0]
        position = (frame.shape[1] - text_width - text_margin , text_top + text_line_height)
        cv2.putText(frame , text = text , org = position , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = fontScale , color = (0 , 255 , 255) , thickness = 1 , lineType = cv2.LINE_AA)

    def show_brakes(frame , brake , target_id , ttc , lane_status):
        target_text = "-" if target_id is None else str(target_id)
        ttc_text = "-" if ttc is None else f"{ttc:.1f}s"
        text = f"BRAKE:{brake:.2f} ID:{target_text} TTC:{ttc_text} LANE:{lane_status}"
        color = (0 , 0 , 255) if brake > 0 else (0 , 255 , 0)
        if brake == 0 and lane_status != "current":
            color = (0 , 180 , 255)
        cv2.putText(frame , text = text , org = (text_margin , text_top) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = fontScale , color = color , thickness = 1 , lineType = cv2.LINE_AA)

    def show_refiner(frame , status , avalible_points):
        for index , side in enumerate(("L" , "R")):
            text = f"REFINER {side}: {status[side]} POINTS:{avalible_points[side]}"
            color = (0 , 255 , 0) if status[side] == "tracking" else (0 , 180 , 255)
            cv2.putText(frame , text = text , org = (text_margin , text_top + (index + 1) * text_line_height) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = fontScale , color = color , thickness = 1 , lineType = cv2.LINE_AA)

    def draw_boxes(detector , frame , trackings , od_colors):
        for track_id , tracking in trackings.items():
            if tracking["status"] != "tracked":
                continue
            x1, y1, x2, y2 = map(int , tracking["box"])
            label = tracking["label"]
            score = tracking["score"]
            direction = tracking["direction"]
            cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)
            cv2.putText(frame , text = detector.class_names[label] + f" {score:.2f} ID:{track_id} {direction}" , org = (x1 , max(y1 - 5 , 0)) , fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale = 0.3 , color = od_colors[label] , thickness = 1 , lineType = cv2.LINE_AA)

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

    def draw_lane_lines(frame , llane_points , rlane_points):
        for lane_points in (llane_points , rlane_points):
            if len(lane_points) < 3:
                continue

            points = np.array(lane_points , dtype = np.int32).reshape(-1 , 1 , 2)
            cv2.polylines(frame , [points] , isClosed = False , color = (0 , 255 , 255) , thickness = 1 , lineType = cv2.LINE_AA)
