import os
import time
from json import dumps, loads
import json
import socket
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from shared.two_way import Dummy
from multiprocessing import Queue
from shared.burst_connection import Msg, Burst_connection
from random import randint

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

class Pub_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.model = None
        self.cfg = None

def main():
    
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    IPAddr = str(IPAddr) + str(random_with_N_digits(5))
    dmy = Pub_BC('agg', IPAddr)
    dmy.comm_metrics['agg'] = -1
    dmy.add_msg_to_q('agg', dmy.QUEUE,dmy.QUEUE, 'init')
    dmy.run()
    
    

if __name__ == '__main__':
    main()

