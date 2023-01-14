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

class Pub_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.model = None
        self.cfg = None
    def process_jobs(self):
        while self.job_q:
            job = self.job_q.pop(0)
            assert (type(job) is dict)
            if job['type'] == 'update_model':
                self.model =model_from_json(job['data'])
                self.add_msg_to_q('agg', self.QUEUE,
                 f"model updated from {self.QUEUE}")
            if job['type'] == 'update_config':
                self.cfg = loads(job['data'])
                self.model.compile(loss=self.cfg['loss_metric'], optimizer=self.cfg['optimizer'])
                self.add_msg_to_q('agg', self.QUEUE,
                 f"config updated from {self.QUEUE}")

    

def main():
    
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    dmy = Pub_BC('agg', IPAddr)

    dmy.add_msg_to_q('agg', IPAddr,IPAddr, 'init')
    dmy.run()
    
    

if __name__ == '__main__':
    main()

