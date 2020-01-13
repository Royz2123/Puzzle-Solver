import serial
from time import sleep

# Enter your COM port in the below line
ard = serial.Serial('COM7', 9600)
sleep(2)
print(ard.readline(ard.inWaiting()))



def send_command(motor, mode):
  cmnd = "%d:%d\n" % (ord(motor[0]), mode)

  print(ard.readline(ard.inWaiting()))
  print(ard.readline(ard.inWaiting()))

  ard.write(cmnd.encode())


# def take_step(motor="X", delay=0.0001):
#   send_command(MOTOR_MAP[motor]["step"], 1)
#   sleep(delay)
#   send_command(MOTOR_MAP[motor]["step"], 0)
#   sleep(delay)
#
#
# def change_dir(motor="X", dir=0):
#   send_command(MOTOR_MAP[motor]["dir"], dir)
#   sleep(0.01)
#
# def control_motor(motor="X", control=0):
#   send_command(MOTOR_MAP[motor]["dir"], control)
#   sleep(0.01)


# def get_log():
#   ard.re
#   data = ard.read_all(ard.inWaiting())
#   print(data)


while True:
  send_command("P", 0)

  sleep(2)

  send_command("P", 1)

  sleep(0.8)

  send_command("L", 0)

  sleep(2)

  send_command("L", 1)

  sleep(0.8)