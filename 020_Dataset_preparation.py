import time
import serial
import json
import csv

dataset_name = 'testing1.csv'

try:
    ser_com = serial.Serial(port='COM3',
                            baudrate=9600,
                            stopbits=1,
                            parity='N',
                            bytesize=8)

except:
    while 1:
        print("Serial Error Occured!")
        sleep(2)

i = 716


with open(dataset_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Raw_value', 'Volts', 'Temperature', 'Humidity'])

for i in range(0, 10):
    data = ser_com.readline().decode().rstrip().replace('\x00', '')
    print(data)

    json_data = json.loads(data)
    print(json_data['Raw_value'])
    print(json_data['Volts'])
    print(json_data['Temperature'])
    print(json_data['Humidity'])

    i = i+1

    with open('household.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([json_data['Raw_value'],
                        json_data['Volts'],
                        json_data['Temperature'],
                        json_data['Humidity']])

    if(i >= 10):
        print("Data Collection Completed!")