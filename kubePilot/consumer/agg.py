import yaml
import pdb
from shared.burst_connection import Msg, Burst_connection
import os
from json import dumps, loads
import json
import random
import tensorflow as tf
import numpy as np
import glob
from json import JSONEncoder
import logging
from shared.utils import IIterable, directory_manager
from random import shuffle
import datetime
from random import randint


def random_with_N_digits(n):
    """
    Generate integer with n digits, useed to create unique queue names
    :param n:
    :return:
    """
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


class Agg_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.raw_model = tf.keras.models.load_model(os.path.join('base-25.h5'))
        gb = glob.glob('data/25/train/*X.npy')
        self.run_metrics_location = directory_manager()
        self.run_data = []
        # limited to data size right now
        self.data_files = [g for g in gb]
        shuffle(self.data_files)
        LOGGER.warning(str(self.data_files))
        with open('config.yaml', 'r') as file:
            self.cfg = yaml.safe_load(file)
        data = self.data_files.pop(0)
        data = data.replace("train", "test")
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.idata = IIterable(x, y, self.cfg['chunk_size'])
        self.round = 0

    def process_metrics(self):
        """
        write metrics misc.json file, appending new dictionary to json list
        """
        if self.round % 10 == 0:
            json_object = None
            with open(os.path.join(self.run_metrics_location, "misc.json"), 'r') as openfile:
                json_object = json.load(openfile)
            dict_ = {"latency": self.pong - self.ping, "avg_msg_size": np.abs(np.mean(self.size_buffer))}
            json_object.append(dict_)
            self.size_buffer = []
            save_file = open(os.path.join(self.run_metrics_location, "misc.json"), "w")
            json.dump(json_object, save_file, indent=6)
            save_file.close()
        self.round += 1


def main():
    dmy = Agg_BC(None, 'agg')
    dmy.run()


if __name__ == '__main__':
    main()
