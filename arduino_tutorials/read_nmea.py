import serial
import time

# store 'N' and 'W' coordinates
list_n = []  # store 'N' coordinates
list_w = []  # store 'W' coordinates

# tracker to determine if last coord was same as latest
track_lat = 0
track_lon = 0

with serial.Serial('COM5', baudrate=115200, timeout=1) as ser:

    # read from the serial output
    for i in range(1000):

        line = ser.readline().decode('ascii', errors='replace')
        # response = line.strip()
        # print(line.strip())
        response = line.split(',')

        if len(response) > 4 and line.find("$GPRMC") == 0:
            print(response)
            lat = ((float(response[3]) / 100.00) - (int(float(response[3]) / 100.00))) / 0.6 + int(
                float(response[3]) / 100.00)
            lon = ((float(response[5]) / 100.00) - (int(float(response[5]) / 100.00))) / 0.6 + int(
                float(response[5]) / 100.00)
            # str(latdegreeConversion) + ' ' + response[4]  # 'N'
            # str(londegreeConversion) + ' ' + response[6]  # 'W'

            list_n.append(lat)
            list_w.append(lon)

            if (lat != track_lat) or (lon != track_lon):

                print('Run image detection and classification')

            track_lat = lat
            track_lon = lon

        time.sleep(.1)
