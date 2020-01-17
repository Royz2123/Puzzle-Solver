import serial
import numpy as np
from time import sleep

# Enter your COM port in the below line
try:
    ard = serial.Serial('COM3', 9600)
    sleep(2)
    print(ard.readline(ard.inWaiting()))
except:
    print("Arduino Disconnected!")

# HOME = (1793 - 1706, 1313)
HOME = (1793 - 1693, 1323) # with stopper
# X_SPEED = 500
# Y_SPEED = 1000
# Z_SPEED = 1000
# THETA_SPEED = 1000
Z_DROP = -400
# ON = 0
# OFF = 1

X_SPEED_FAST = 2000
X_ACC_FAST = 4000  # 5000
X_SPEED_SLOW = 1000
X_ACC_SLOW = 2500
Y_SPEED_FAST = 1000  # 2000
Y_ACC_FAST = 2000  # 4000
THETA_SPEED = 250
THETA_ACC = 1000
Z_SPEED = 1000
Z_ACC = 2000


def send_command(motor, mode, speed=1000):
    cmnd = "%d:%d:%d\n" % (ord(motor[0]), mode, speed)

    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))

    ard.write(cmnd.encode())
    sleep(0.01)


def send_command_accel(x_steps=0, y_steps=0, theta_steps=0, z_steps=0, pump_on=0, l_on=0,
                       x_speed=X_SPEED_FAST, y_speed=Y_SPEED_FAST, theta_speed=THETA_SPEED, z_speed=Z_SPEED,
                       x_acc=X_ACC_FAST, y_acc=Y_ACC_FAST, theta_acc=THETA_ACC, z_acc=Z_ACC):
    cmnd = "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n" % (x_steps, y_steps, theta_steps, z_steps, pump_on, l_on,
                                                            x_speed, y_speed, theta_speed, z_speed,
                                                            x_acc, y_acc, theta_acc, z_acc)

    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))

    ard.write(cmnd.encode())
    sleep(0.1)


def pixels2steps(motor, pixels):
    if motor == "X":
        # return int(1000 / 916.081 * pixels)
        # return int(1000 / 903 * pixels)
        return int(1000 / 908 * pixels)
    if motor == "Y":
        # return int(1000 / 866 * pixels)
        # return int(1000 / 899 * pixels)
        return int(500 / 438 * pixels)


def radians2steps(theta):
    return int(200 / (2 * np.pi) * theta)


# def home(coords):
#     """
#     Homes in on HOME using the current coordinates, rams the arm
#     :param coords: (x,y) coords
#     """
#     send_command("X", pixels2steps("X", HOME[0] - coords[0]) - 200, X_SPEED)
#     sleep(2)
#     send_command("Y", pixels2steps("Y", HOME[1] - coords[1]) + 200, Y_SPEED)
#     sleep(2)


def home_accel(coords):
    """
        Homes in on HOME using the current coordinates, rams the arm
        :param coords: (x,y) coords
        """
    send_command_accel(x_steps=pixels2steps("X", HOME[0] - coords[0]) - 200,
                       y_steps=pixels2steps("Y", HOME[1] - coords[1]) + 200,
                       x_speed=X_SPEED_SLOW, x_acc=X_ACC_SLOW)
    sleep(3)


# def execute_command(coords_arr):
#     """
#     Executes entire sequence of moves, homing to (0,0) after every complete
#     placement.
#     Assumes arm is homed on HOMING constant.
#     :param coords_arr: array of coords, [(x1,y1,x2,y2,theta),...]
#     """
#
#     home((4000, 1311))
#     sleep(5)
#     for tup in coords_arr:
#         # print(tup)
#         # move to piece to be picked-up
#         send_command("X", pixels2steps("X", 50), 3000)
#         send_command("X", pixels2steps("X", tup[0] - HOME[0] - 50), X_SPEED)
#         sleep(2)
#         send_command("Y", pixels2steps("Y", tup[1] - HOME[1]), Y_SPEED)
#         sleep(2)
#         send_command("Z", Z_DROP, Z_SPEED)
#         sleep(2)
#         send_command("P", ON)
#         sleep(2)
#         send_command("Z", -Z_DROP, Z_SPEED)
#         sleep(2)
#         # move to placement and place
#         send_command("X", pixels2steps("X", tup[2] - tup[0]), X_SPEED)
#         sleep(2)
#         send_command("Y", pixels2steps("Y", tup[3] - tup[1]), Y_SPEED)
#         sleep(2)
#         send_command("R", radians2steps(tup[4]), THETA_SPEED)
#         sleep(2)
#         send_command("Z", Z_DROP, Z_SPEED)
#         sleep(2)
#         send_command("P", OFF)
#         sleep(2)
#         send_command("Z", -Z_DROP, Z_SPEED)
#         sleep(2)
#
#         # home
#         home((tup[2], tup[3]))


def execute_command_accel(coords_arr):
    """
    Executes entire sequence of moves, homing to (0,0) after every complete
    placement.
    Assumes arm is homed on HOMING constant.
    :param coords_arr: array of coords, [(x1,y1,x2,y2,theta),...]
    """

    home_accel((4000, 1311))
    sleep(5)
    for tup in coords_arr:
        # print(tup)
        # move to piece to be picked-up
        send_command_accel(x_steps=pixels2steps("X", tup[0] - HOME[0]))
        sleep(3)
        send_command_accel( y_steps=pixels2steps("Y", tup[1] - HOME[1]))
        sleep(3)
        send_command_accel(theta_steps=-radians2steps(tup[4]) // 2, pump_on=1)
        sleep(3)
        send_command_accel(z_steps=Z_DROP, pump_on=1)
        sleep(3)
        send_command_accel(z_steps=-Z_DROP, pump_on=1)
        sleep(3)
        # move to placement and place
        send_command_accel(x_steps=pixels2steps("X", tup[2] - tup[0]), pump_on=1)
        sleep(3)
        send_command_accel( y_steps=pixels2steps("Y", tup[3] - tup[1]),pump_on=1)
        sleep(3)
        send_command_accel(theta_steps=radians2steps(tup[4]), pump_on=1)
        sleep(3)
        send_command_accel(z_steps=Z_DROP, pump_on=1)
        sleep(3)
        send_command_accel(z_steps=-Z_DROP, pump_on=0)
        sleep(3)

        # home
        send_command_accel(theta_steps=-radians2steps(tup[4]) // 2)
        sleep(3)
        home_accel((tup[2], tup[3]))


def get_log():
    data = ard.read_all(ard.inWaiting())
    print(data)


if __name__ == "__main__":
    #     home_accel((4000,1311))
    send_command_accel(pump_on=0)
    # send_command("R",radians2steps(np.pi/2),1000)
#     send_command("Z",400,2000)
# sleep(5)
# send_command_accel(z_steps=300,pump_on=1)
# send_command("R",radians2steps(np.pi/2),500)
# send_command("X",3000,500)
# sleep(2)
# send_command("X",-3000,500)
# for i in range(15):
#     send_command("Y",300,1000)
#     sleep(1)
#     send_command("Y", -300, 1000)
#     sleep(1)

# above = camera.take_picture(still_camera=True)
# above = util.relevant_section(above)
# above = cv2.resize(above,None, fx=0.2,fy=0.2)
# cv2.imshow("above", above)
# cv2.waitKey(0)
# circles = cv2.HoughCircles(above,cv2.HOUGH_GRADIENT,1,20)
# circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(above, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(above, (i[0], i[1]), 2, (0, 0, 255), 3)
#
# cv2.imshow('detected circles', above)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# send_command("X",-50,X_SPEED)
# send_command("Z", -200)
# send_command("P", 0)
# sleep(1)
# send_command("Z", 200)
#
# send_command("X", 1500)
# send_command("Y", -1000)
#
# send_command("Z", -400)
# send_command("P", 1)
# sleep(1)
# send_command("Z", 270)
#
# send_command("X", -1500)
# send_command("Y", 1000)
