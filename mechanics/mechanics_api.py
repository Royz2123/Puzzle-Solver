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

HOME = (0, 0)
X_SPEED = 500
Y_SPEED = 1000
Z_SPEED = 500
THETA_SPEED = 500
Z_DROP = 250
ON = 0
OFF = 1


def send_command(motor, mode, speed=1000):
    cmnd = "%d:%d:%d\n" % (ord(motor[0]), mode, speed)

    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))

    ard.write(cmnd.encode())
    sleep(0.01)


def pixels2steps(motor, pixels):
    if motor == "X":
        return int(1000 / 916.081 * pixels)
    if motor == "Y":
        return int(1000 / 866 * pixels)


def radians2steps(theta):
    return int(200 / (2 * np.pi) * theta)


def home(coords):
    """
    Homes in on HOME using the current coordinates, rams the arm
    :param coords: (x,y) coords
    """
    send_command("X", HOME[0] - (coords[0]+200), X_SPEED)
    send_command("Y", HOME[1] - (coords[1]+200), Y_SPEED)


def execute_command(coords_arr):
    """
    Executes entire sequence of moves, homing to (0,0) after every complete placement.
    Assumes arm is homed on HOMING constant.
    :param coords_arr: array of coords, [(x1,y1,x2,y2,theta),...]
    """
    for tup in coords_arr:
        # move to piece to be picked-up
        send_command("X", pixels2steps("X", tup[0]), X_SPEED)
        send_command("Y", pixels2steps("Y", tup[1]), Y_SPEED)
        send_command("Z", Z_DROP, Z_SPEED)
        send_command("P", ON)
        sleep(0.1)
        send_command("Z", -Z_DROP, Z_SPEED)

        # move to placement and place
        send_command("X", pixels2steps("X", tup[2] - tup[0]), X_SPEED)
        send_command("Y", pixels2steps("Y", tup[3] - tup[1]), Y_SPEED)
        send_command("R", radians2steps(tup[4]), THETA_SPEED)
        send_command("Z", Z_DROP, Z_SPEED)
        send_command("P",OFF)
        sleep(0.1)
        send_command("Z", -Z_DROP, Z_SPEED)

        # home
        home((tup[2],tup[3]))


def get_log():
    data = ard.read_all(ard.inWaiting())
    print(data)


if __name__ == "__main__":
    above = camera.take_picture(still_camera=True)
    above = util.relevant_section(above)
    above = cv2.resize(above,None, fx=0.2,fy=0.2)
    cv2.imshow("above", above)
    cv2.waitKey(0)
    circles = cv2.HoughCircles(above,cv2.HOUGH_GRADIENT,1,20)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(above, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(above, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', above)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
