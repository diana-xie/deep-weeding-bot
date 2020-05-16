"""
https://makersportal.com/blog/2018/2/25/python-datalogger-reading-the-serial-output-from-arduino-to-analyze-data-using-pyserial
"""

import serial
ser = serial.Serial('COM5')
ser.flushInput()

while True:
    try:
        # ser_bytes = ser.readline().decode("ascii", "ignore")
        ser_bytes = ser.readline().decode('ascii', errors='replace')
        # decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
        # decoded_bytes = ser.readline().decode("ascii", "ignore")
        # print(decoded_bytes)
        print(ser_bytes)
    except:
        print("Keyboard Interrupt")
        break