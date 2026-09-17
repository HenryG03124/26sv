import vgamepad as vg
import time

class InputController:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()

    def steering_controller(self , steering):
        steering = max(-1.0 , min(1.0 , steering))
        self.gamepad.left_joystick_float(x_value_float = steering , y_value_float = 0.0)

        self.gamepad.update()

        time.sleep(0.01)

    def close(self):
        self.gamepad.reset()