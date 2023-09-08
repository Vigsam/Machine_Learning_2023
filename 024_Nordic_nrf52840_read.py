import serial
import json
from time import sleep

def try_serial():
    try:
        serial_read = serial.Serial(port='COM3',
                                    baudrate=9600,
                                    bytesize=8,
                                    parity='N',
                                    stopbits=1)

    except:
        print("Something wrong with the serial port")


try:
    serial_read = serial.Serial(port='COM3',
                                baudrate=9600,
                                bytesize=8,
                                parity='N',
                                stopbits=1)

    print("Serial Port Connected!")

except:
    print("Something wrong with the serial port")

while(1):
    read_data = serial_read.readline().decode().rstrip().replace('\x00', '')
    print(read_data)

    try:

        json_data = json.loads(read_data)

        '''
        
        TIVA C
        
        print(json_data["ADC Value"])
        print(json_data["voltage"])
        print(json_data["Status"])
    
        '''

        print(json_data["Temperature"])
        print(json_data["Humidity"])
        print(json_data["Red"])
        print(json_data["Green"])
        print(json_data["Blue"])

    except:
        print("Try in Next Read")

