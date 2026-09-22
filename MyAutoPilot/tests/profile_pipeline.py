"""Read-only timing: no gamepad, no game launch, no display window.

python -B -m MyAutoPilot.tests.profile_pipeline --capture --frames 40
python -B -m MyAutoPilot.tests.profile_pipeline --image path/to/frame.jpg
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from MyAutoPilot.screen_capture import ScreenCapture
from MyAutoPilot.pixel_classifier_v2 import PixelClassifierV2
from MyAutoPilot.object_detector import ObjectDetectorLite
from MyAutoPilot.mov_detector import MovDetector
from MyAutoPilot.lane_refiner import LaneRefiner
from MyAutoPilot.steering import Steering
from MyAutoPilot.auto_brakes import AutoBrakes
from MyAutoPilot.draw import Draw
from MyAutoPilot.logger import Logger
from MyAutoPilot.scs_telemetry import SCSTelemetry


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--capture" , action = "store_true")
    parser.add_argument("--image")
    parser.add_argument("--frames" , type = int , default = 40)
    parser.add_argument("--monitor" , type = int , default = 1)
    args = parser.parse_args()
    capture = ScreenCapture(args.monitor) if args.capture else None
    frame = cv2.imread(args.image) if args.image else np.zeros((2160 , 3840 , 3) , dtype = np.uint8)
    classifier = PixelClassifierV2()
    detector = ObjectDetectorLite()
    root = Path(__file__).resolve().parents[1]
    classifier.load_model(root / "pc_model_v2_bdd100k.pth")
    detector.load_model(root / "od_model_Lite.pth")
    refiner = LaneRefiner(188 , 292 , 10)
    movement = MovDetector()
    brakes = AutoBrakes()
    reader = SCSTelemetry()
    logger = Logger(Path(tempfile.mkdtemp(prefix = "autopilot_profile_")))
    measurements = []
    previous = 0
    try:
        for index in range(args.frames + 8):
            timings = {}
            start = last = time.perf_counter()

            def mark(name):
                nonlocal last
                now = time.perf_counter()
                timings[name] = (now - last) * 1000
                last = now

            if capture:
                frame = capture.capture_frame()
            mark("capture")
            timestamp = time.monotonic()
            telemetry = reader.read()
            mark("telemetry")
            mask = classifier.predict(frame , 0.7)
            mark("classifier")
            positions , boxes , scores , labels = detector.predict(frame)
            mark("detector_predict")
            boxes , scores , labels = detector.box_filter(positions , boxes , scores , labels , 640 , 352 , mask , 0.5)
            trackings = movement.update(boxes , scores , labels , timestamp , 70)
            mark("filter_tracking")
            llane_sample_points , rlane_sample_points , llane_points , rlane_points = refiner.refine(mask , 2 , 10 , return_sample = True)
            center = refiner.calculate_center_points(llane_points , rlane_points)
            mark("refiner")
            previous = Steering.steering(center , 188 , 3 , previous , telemetry["speed_kmh"] , timestamp , 292)
            brake = brakes.brake(llane_points , rlane_points , trackings , timestamp , 352 , 640)
            mark("steering_brake")
            display = cv2.resize(frame , (640 , 352))
            lane_mask = np.zeros_like(mask)
            if llane_points and rlane_points:
                cv2.fillPoly(lane_mask , [np.array(llane_points + rlane_points[::-1] , dtype = np.int32)] , 1)
            colors = np.array([[0 , 0 , 0] , [0 , 255 , 0] , [0 , 0 , 255] , [0 , 255 , 255]] , dtype = np.uint8)
            active = ((mask == 1) & (lane_mask > 0)) | (mask == 2) | (mask == 3)
            blended = cv2.addWeighted(display , 0.7 , colors[mask] , 0.3 , 0)
            display[active] = blended[active]
            Draw.draw_boxes(detector , display , trackings , ((255 , 64 , 64) , (64 , 128 , 255)))
            Draw.draw_lane_points(display , llane_sample_points , rlane_sample_points)
            Draw.draw_lane_lines(display , llane_points , rlane_points)
            Draw.draw_road_center(display , center , 188 , 292)
            Draw.draw_refiner(display , refiner.status , refiner.avalible_points)
            Draw.draw_brake(display , brake , brakes.target_id , brakes.ttc , brakes.lane_status)
            display = cv2.resize(display , (1920 , 1080))
            Draw.show_fps(display , 0)
            Draw.show_speed(display , telemetry)
            mark("draw_resize_no_gui")
            logger.write(timestamp , {"center_points": center , "llane_points": llane_points , "rlane_points": rlane_points ,
                                      "lane_sample_points": {"L": llane_sample_points , "R": rlane_sample_points} ,
                                      "trackings": trackings , "telemetry": telemetry , "steering": previous})
            mark("logging")
            timings["total_no_control_no_gui"] = (time.perf_counter() - start) * 1000
            if index >= 8:
                measurements.append(timings)
        result = {key: {"mean_ms": round(float(np.mean([v[key] for v in measurements])) , 3) ,
                        "p90_ms": round(float(np.percentile([v[key] for v in measurements] , 90)) , 3)}
                  for key in measurements[0]}
        print(json.dumps({"torch_threads": torch.get_num_threads() , "cv_threads": cv2.getNumThreads() ,
                          "input_size": list(frame.shape) , "timings": result} , indent = 2))
    finally:
        if capture:
            capture.close()
        reader.close()
        logger.close()


if __name__ == "__main__":
    main()
