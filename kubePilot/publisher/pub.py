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

def main():
    
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    dmy = Burst_connection('agg', IPAddr)
    dmy.run()
    
    

if __name__ == '__main__':
    main()

