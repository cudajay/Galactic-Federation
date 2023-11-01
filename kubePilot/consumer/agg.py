import sys
sys.path.append('/app/src/shared/')
import yaml
import pdb
from burst_connection import Msg, Burst_connection
from rule_engines import Base_Engine, GT_fedAvg_Engine, GT_fedsgd_engine
import os
from json import dumps, loads
import json
import random
import tensorflow as tf
import numpy as np
import glob
from json import JSONEncoder
import logging
from utils import IIterable, directory_manager, parse_init_config, tf_loader
from random import shuffle
import datetime
from random import randint
import argparse

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
    def __init__(self, writeto: str, consumefrom: str, expn, cfg):
        super().__init__(writeto, consumefrom)
        self.cfg = cfg
        self.raw_model = tf_loader(tf.keras.models.load_model(self.cfg['base_model']), self.cfg)
        self.raw_model.compile(loss=self.cfg['loss_metric'], optimizer=self.cfg['optimizer'])     
        self.run_metrics_location = directory_manager(self.cfg, expn)
        self.run_data = []
        # limited to data size right now
        gb = glob.glob('/app/data/25/train/*X.npy')
        self.data_files = [g for g in gb]
        shuffle(self.data_files)
        LOGGER.warning(str(self.data_files))
        data = self.data_files.pop(0)
        data = data.replace("train", "test")
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.idata = IIterable(x, y, self.cfg['chunk_size'])
        self.round = 0
        self.C = []
        if self.cfg['re'] == 'fedAvg':
            self.re = GT_fedAvg_Engine(self)
        if self.cfg['re'] == 'fedSgd':
            self.re = GT_fedsgd_engine(self)
            
        self.g_min = np.Inf
        self.patience_test = 0
        self.last_target_score = None
        self.exp_n = 0

    def process_metrics(self):
        """
        write metrics misc.json file, appending new dictionary to json list
        """
        if self.round % 10 == 0:
            json_object = None
            with open(os.path.join(self.run_metrics_location, "misc.json"), 'r') as openfile:
                json_object = json.load(openfile)
            dict_ = {"latency": np.abs(self.pong - self.ping), "avg_msg_size": np.mean(self.size_buffer), 'n_messages': self.msg_id * 2, 'killed': self.comms_enabled,
                     'last_updated': str(datetime.datetime.now())}
            json_object.append(dict_)
            self.size_buffer = []
            save_file = open(os.path.join(self.run_metrics_location, "misc.json"), "w")
            json.dump(json_object, save_file, indent=6)
            save_file.close()
        self.round += 1


def main():
    #raw_model = tf.keras.models.load_model(os.path.join("/app","src", "consumer", 'base-25.h5'))
    parser = argparse.ArgumentParser(description="Debug flag")
    
    # Define positional arguments with default values
    parser.add_argument("--debug", type=int, default=0, help="debug flag, true to debug", required=False)
    args = parser.parse_args()
    with open('/app/src/consumer/config.yaml', 'r') as file:
         cfg = yaml.safe_load(file)

    if cfg['debug'] and not args.debug:
        LOGGER.warning("\n\n\n\n ****In DEBUG LOOP****\n\n\n")
        while True:
            pass
    n_exps = 5
    param_list = parse_init_config(cfg, n_exps)
    LOGGER.warning(str(param_list))
    for i,cfgi in zip(range(n_exps), param_list):
        agg = Agg_BC(None, 'agg'+str(i),i, cfgi)
        agg.run()
        agg.publish_queue()
        LOGGER.warning(f"****\n\n\n\n Study {i} Complete target val: {agg.g_min}")
    LOGGER.warning("****\n\n\n\n Experiment Complete")

if __name__ == '__main__':
    main()
