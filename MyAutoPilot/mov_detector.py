import numpy as np

car_label_id = 1
pedestrian_label_id = 0
max_missing_time = 0.5 #seconds

class MovDetector():
    def __init__(self):
        self.reset()

    def reset(self):
        self.trackings = {}
        self.empty_ids = [i for i in range(200)]

    def match_centers(self , centers , labels , timestamp , spd):
        candidates = []
        for track_id , tracking in self.trackings.items():
            timestamp_delta = timestamp - tracking["last_seen"]
            radius = timestamp_delta * spd
            distances = np.linalg.norm(centers - tracking["center"] , axis = 1)
            selected = (distances <= radius) & (labels == tracking["label"])
            for box_index in np.where(selected)[0]:
                candidates.append((float(distances[box_index]) , track_id , int(box_index)))

        
        matches = {}
        matched_tracks = set()
        for distance , track_id , box_index in sorted(candidates):
            if track_id not in matched_tracks and box_index not in matches:
                matches[box_index] = track_id
                matched_tracks.add(track_id)

        return matches

    def update(self , boxes , scores , labels , timestamp , spd):
        boxes = np.asarray(boxes , dtype = np.float64).reshape(-1 , 4)
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        selected = (labels == car_label_id) | (labels == pedestrian_label_id)
        boxes , scores , labels = boxes[selected] , scores[selected] , labels[selected]
        centers = (boxes[: , : 2] + boxes[: , 2 :]) / 2

        for track_id in list(self.trackings):
            tracking = self.trackings[track_id]
            if timestamp - tracking["last_seen"] > max_missing_time:
                del self.trackings[track_id]
                self.empty_ids.append(track_id)
                self.empty_ids.sort()
            else:
                tracking["status"] = "lost"
                tracking["direction"] = "unknown"

        matches = self.match_centers(centers , labels , timestamp , spd)

        for box_index in range(len(boxes)):
            direction = "unknown"
            if box_index in matches:
                track_id = matches[box_index]
                delta_x , delta_y = centers[box_index] - self.trackings[track_id]["center"]
                if delta_x == 0 and delta_y == 0:
                    direction = "stationary"
                elif abs(delta_y) >= abs(delta_x):
                    direction = "forward" if delta_y < 0 else "backward"
                else:
                    direction = "right" if delta_x > 0 else "left"
            else:
                track_id = self.empty_ids[0] #new object
                self.empty_ids = self.empty_ids[1 : ]

            self.trackings[track_id] = {
                "box": boxes[box_index].tolist() ,
                "center": centers[box_index].tolist() ,
                "score": float(scores[box_index]) ,
                "label": int(labels[box_index]) ,
                "last_seen": timestamp ,
                "direction": direction ,
                "status": "tracked" ,
            }

        return self.trackings


