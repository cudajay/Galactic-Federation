import pika
import yaml
from shared.two_way import Dummy
from shared.burst_connection import Msg, Burst_connection
import os
import time
from json import dumps, loads
import json
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
import glob
from json import JSONEncoder
import logging
from multiprocessing import Queue
from shared.utils import IIterable
from random import shuffle

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

class Agg_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.raw_model = tf.keras.models.load_model(os.path.join('base-25.h5'))
        gb = glob.glob('data/25/train/*X.npy')
        #limited to data size right now
        self.data_files = [g for g in gb]
        shuffle(self.data_files)
        LOGGER.warning(str(self.data_files))
        with open('config.yaml', 'r') as file:
            self.cfg = dumps(yaml.safe_load(file))
        data = self.data_files.pop(0)
        data = data.replace("train", "test")
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.idata = IIterable(x, y, self.cfg['chunk_size'])


def main():
    dmy = Agg_BC(None, 'agg')
    dmy.run()

if __name__ == '__main__':
    main()
