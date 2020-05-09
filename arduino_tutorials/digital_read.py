""" https://realpython.com/arduino-python/ """

import pyfirmata
import time

board = pyfirmata.Arduino('/COM5')

it = pyfirmata.util.Iterator(board)
it.start()

board.digital[2].mode = pyfirmata.INPUT

while True:
    sw = board.digital[2].read()
    if sw is True:
        # print('asdf')
        board.digital[13].write(1)
    else:
        board.digital[13].write(0)
    time.sleep(0.1)