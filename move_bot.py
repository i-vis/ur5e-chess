import time

# import serial
import socket
import numpy as np
import lib_gripper


class URMover:
    OFFSET = np.float32([0.2975, 0.045, 0.100, 2.191, 2.258, 0.048])
    OFFSET = np.float32([0.2975, 0.047, 0.100, 2.191, 2.258, 0.048])
    SQ_CENTER_OFFSET = np.float32([0, 0, 0, 0, 0, 0])
    SQUARE_SIZE = np.float32([0.035, 0.035, 0, 0, 0, 0])
    GRIP_HEIGHT = 0.05
    GRIP_POS = 0.024
    move_grip_function = """
def myFun():
    i = 0
    movel(%s, a=1, v=1)
    sleep(0.5)
    movel(%s, a=1, v=1)
    sleep(0.5)
    sleep(%d)
    movel(%s, a=1, v=1)
    sleep(0.5)
    movel(%s, a=1, v=1)
    sleep(0.5)
    movel(%s, a=1, v=1)
    sleep(0.5)
    sleep(%d)
    movel(%s, a=1, v=1)
    sleep(0.5)
    movel(%s, a=1, v=1)
    sleep(0.5)
end

myFun()
    """

    s: socket.socket
    # gripper : serial.Serial
    gripper : lib_gripper.RG2

    def __init__(self, IP, PORT=30002):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((IP, PORT))
        rg_id = 0
        self.gripper = lib_gripper.RG2(IP, rg_id)
        # self.gripper = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)

    def _sq2cb_pos(self, pos: str) -> str:
        a = [8 - int(pos[1]), (ord(pos[0]) - ord("a"))]
        print(a)
        final_pos = (
            self.OFFSET
            + self.SQ_CENTER_OFFSET
            + [a[0] * self.SQUARE_SIZE[0], a[1] * self.SQUARE_SIZE[1], 0, 0, 0, 0]
        )
        res = f"p[{final_pos[0]}, {final_pos[1]}, {final_pos[2]}, {final_pos[3]}, {final_pos[4]}, {final_pos[5]}]"
        # print(res)
        return res

    def move_grip(self, sq1_name, sq2_name, h1, h2):
        pos1 = self._sq2cb_pos(sq1_name)
        pos1_lower = ",".join(
            pos1.split(",")[:2] + [" " + str(h1)] + pos1.split(",")[3:]
        )
        pos2 = self._sq2cb_pos(sq2_name)
        pos2_lower = ",".join(
            pos2.split(",")[:2] + [" " + str(h2)] + pos2.split(",")[3:]
        )
        neutral = self._sq2cb_pos("Y4")
        self.s.send(
            (
                self.move_grip_function
                % (pos1, pos1_lower, 5, pos1, pos2, pos2_lower, 5, pos2, neutral)
            ).encode("utf-8")
        )
        print(
            sq1_name,
            sq2_name,
            self.move_grip_function
            % (pos1, pos1_lower, 5, pos1, pos2, pos2_lower, 5, pos2, neutral),
        )
        # self.s.send(
        #     (
        #         self.move_grip_function
        #         % (pos1_lower, pos1) + "\n").encode("utf-8")
        # )
        # print(
        #         self.move_grip_function
        #         % (pos1, pos1_lower)
        # )
        self.gripper.rg_grip(30, 40)
        time.sleep(6)
        self.gripper.rg_grip(0, 40)
        time.sleep(6)
        # release
        self.gripper.rg_grip(30, 40)
