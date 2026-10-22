import math

target_speed = 50 #km/h
Kp = 0.1
Ki = 0.02

class Throttle():
    def __init__(self):
        self.integral = 0.0
        self.prev_time = None
        self.target_speed = target_speed

    def throttle(self , telemetry , center_tracking_frames , brake , timestamp):
        speed = telemetry["speed_kmh"]
        if center_tracking_frames < 5 or brake > 0 or telemetry["status"] != "ok" or speed is None or not math.isfinite(speed) or speed < 0:
            self.integral = 0.0
            self.prev_time = None
            return 0.0

        dt = 0.0 if self.prev_time is None else max(0.0 , min(timestamp - self.prev_time , 0.5))
        self.prev_time = timestamp
        error = self.target_speed - speed
        throttle = Kp * error + self.integral

        if 0 < throttle < 1 or (throttle >= 1 and error < 0) or (throttle <= 0 and error > 0):
            self.integral = max(0.0 , min(1.0 , self.integral + Ki * error * dt))

        return max(0.0 , min(1.0 , Kp * error + self.integral))

    def update_target(self , set):
        if set == "up":
            self.target_speed += 5
        else:
            self.target_speed -= 5

    def get_target(self):
        return self.target_speed
