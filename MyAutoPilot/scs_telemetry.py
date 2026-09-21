import ctypes
import math
import os
import struct
import time
from ctypes import wintypes

# truckermudgeon/scs-sdk-plugin, revision 12, scs-telemetry-common.hpp.
# Protocol reference: c8910d6e9a5ca0c2fa018942054263292cab19a4.
memory_name = "Local\\SCSTelemetry"
snapshot_size = 2248
stale_interval = 0.5 #seconds

class SCSTelemetry():
    def __init__(self):
        self.handle = None
        self.address = None
        self.next_connect = 0
        self.prev_timestamp = None
        self.last_update = None
        self.kernel = None
        if os.name == "nt":
            self.kernel = ctypes.WinDLL("kernel32" , use_last_error = True)
            self.kernel.OpenFileMappingW.argtypes = [wintypes.DWORD , wintypes.BOOL , wintypes.LPCWSTR]
            self.kernel.OpenFileMappingW.restype = wintypes.HANDLE
            self.kernel.MapViewOfFile.argtypes = [wintypes.HANDLE , wintypes.DWORD , wintypes.DWORD , wintypes.DWORD , ctypes.c_size_t]
            self.kernel.MapViewOfFile.restype = ctypes.c_void_p
            self.kernel.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
            self.kernel.UnmapViewOfFile.restype = wintypes.BOOL
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.restype = wintypes.BOOL

    def connect(self):
        if self.kernel is None:
            return
        # Open existing memory read-only. Never create a fake SDK mapping.
        self.handle = self.kernel.OpenFileMappingW(4 , False , memory_name)
        if self.handle:
            self.address = self.kernel.MapViewOfFile(self.handle , 4 , 0 , 0 , snapshot_size)
            if not self.address:
                self.close()

    @staticmethod
    def decode(snapshot):
        revision = struct.unpack_from("<I" , snapshot , 40)[0]
        if revision != 12:
            return {"status": "unsupported" , "revision": revision , "speed_kmh": None}
        timestamp , simulated_time = struct.unpack_from("<QQ" , snapshot , 8)
        speed , rpm , user_steer , throttle , brake , clutch , game_steer = struct.unpack_from("<7f" , snapshot , 948)
        position = struct.unpack_from("<3d" , snapshot , 2200)
        rotation = struct.unpack_from("<3d" , snapshot , 2224)
        velocity = struct.unpack_from("<3f" , snapshot , 1868)
        angular_velocity = struct.unpack_from("<3f" , snapshot , 1880)
        wheel_count = min(struct.unpack_from("<I" , snapshot , 80)[0] , 16)
        wheels = struct.unpack_from("<16f" , snapshot , 1200)
        active = bool(snapshot[0])
        paused = bool(snapshot[4])
        return {
            "status": "inactive" if not active else "paused" if paused else "ok" ,
            "revision": revision , "sdk_active": active , "paused": paused ,
            "sdk_timestamp_us": timestamp , "simulated_time_us": simulated_time ,
            "speed_mps": speed , "speed_kmh": speed * 3.6 if active and not paused else None ,
            "engine_rpm": rpm , "gear": struct.unpack_from("<i" , snapshot , 504)[0] ,
            # SDK steer is positive LEFT; joystick commands are positive RIGHT.
            "user_steer": user_steer , "game_steer": game_steer ,
            "user_steer_right": -user_steer , "game_steer_right": -game_steer ,
            "throttle": throttle , "brake": brake , "clutch": clutch ,
            "game_throttle": struct.unpack_from("<f" , snapshot , 976)[0] ,
            "game_brake": struct.unpack_from("<f" , snapshot , 980)[0] ,
            "position_m": list(position) , "rotation_turns": list(rotation) ,
            "local_velocity_mps": list(velocity) ,
            "yaw_rate_ccw_rad_s": angular_velocity[1] * 2 * math.pi ,
            "wheel_steering_deg": [wheels[i] * 360 for i in range(wheel_count)] ,
        }

    def read(self):
        now = time.monotonic()
        if not self.address and now >= self.next_connect:
            self.connect()
            self.next_connect = now + 1
        if not self.address:
            return {"status": "missing" , "speed_kmh": None , "read_timestamp": now}
        data = self.decode(ctypes.string_at(self.address , snapshot_size))
        data["read_timestamp"] = now
        if data["status"] == "unsupported":
            self.close()
            return data
        timestamp = data["sdk_timestamp_us"]
        if timestamp != self.prev_timestamp:
            self.prev_timestamp = timestamp
            self.last_update = now
        data["age_s"] = now - self.last_update
        if data["status"] == "ok" and data["age_s"] > stale_interval:
            data["status"] = "stale"
            data["speed_kmh"] = None
        if data["status"] in ("inactive" , "stale"):
            self.close()
            self.next_connect = now + 1
        return data

    def close(self):
        if self.address:
            self.kernel.UnmapViewOfFile(self.address)
            self.address = None
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
