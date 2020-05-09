""" https://realpython.com/arduino-python/ """

import pyfirmata
import time

board = pyfirmata.Arduino('/COM5')

while True:
    board.digital[13].write(1)
    time.sleep(1)
    board.digital[13].write(0)
    time.sleep(1)