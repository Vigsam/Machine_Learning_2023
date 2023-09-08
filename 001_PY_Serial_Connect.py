import serial
from time import sleep

serial_communication = serial.Serial(port='COM11',
                                     baudrate=9600,
                                     bytesize=8,
                                     parity='N',
                                     stopbits=1)

while(1):
    line_read = serial_communication.readline().decode().rstrip().replace("\x00", "")
    print(line_read)