import serial
from time import sleep

# Enter your COM port in the below line
try:
    ard = serial.Serial('COM7', 9600)
    sleep(2)
    print(ard.readline(ard.inWaiting()))
except:
    print("Arduino Disconnected!")


def send_command(motor, mode, speed=1000):
    cmnd = "%d:%d:%d\n" % (ord(motor[0]), mode, speed)

    print(ard.readline(ard.inWaiting()))
    print(ard.readline(ard.inWaiting()))

    ard.write(cmnd.encode())
    sleep(1)


def get_log():
    data = ard.read_all(ard.inWaiting())
    print(data)


if __name__ == "__main__":
    send_command("Z", -200)
    send_command("P", 0)
    sleep(1)
    send_command("Z", 200)

    send_command("X", 1500)
    send_command("Y", -1000)

    send_command("Z", -400)
    send_command("P", 1)
    sleep(1)
    send_command("Z", 270)

    send_command("X", -1500)
    send_command("Y", 1000)
