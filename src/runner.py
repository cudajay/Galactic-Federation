import yaml
import multiprocessing as mp
import time
import random
from tensorflow import keras
import numpy as np
from Orbiters import LocalOrbiter
import logging

logging.basicConfig(filename="log.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')


def get_fed_avg(g, ws):
    n_layers = len(ws[0].trainable_weights)
    agg_layers = []
    for i in range(n_layers):
        tns = k = keras.backend.zeros(ws[0].trainable_weights[i].shape)
        for w in ws:
            tns = tns + w.trainable_weights[i]
        g.trainable_weights[i].assign(tns / len(ws))
    return agg_layers


def magistrate(pipes, cfg):
    # Broadcast base model to all K
    mdls = []
    preds = ''
    global_model = keras.models.load_model(f'../models/A-1.h5')
    for p in pipes:
        p.send({f'type': 'update_model', 'content': f'../models/A-1.h5'})
        p.send({'type': 'update_data', 'content': '../'})
        p.send({'type': 'train_model'})
    while pipes:
        for p in pipes:
            if p.poll():
                msg = p.recv()
                if msg['type'] == 'msg':
                    print(msg['content'])
                if msg['type'] == 'preds':
                    preds = preds + "\n" + msg['content']
                if msg['type'] == 'weights':
                    mdls.append(msg['content'])
                if msg['type'] == 'close_pipe':
                    pipes.remove(p)
        if len(mdls) >= 3:
            get_fed_avg(global_model ,mdls)
            global_model.save("tmp.h5")
            for p in pipes:
                p.send({f'type': 'update_model', 'content': f'tmp.h5'})
                p.send({'type': 'train_model'})
            mdls = []
    print(preds)


def main():
    pipes = []
    orbiters = []
    orb_jobs = []
    logger = logging.getLogger()
    # log all messages, debug and up
    logger.setLevel(logging.INFO)
    for i in range():
        x, y = mp.Pipe()
        pipes.append(x)
        orb = LocalOrbiter(i)
        orb_jobs.append(mp.Process(target=orb.orbit, args=(y, 'f',)))
        orbiters.append(orb)

    mj = mp.Process(target=magistrate, args=(pipes, 'f',))
    mj.start()
    for o in orb_jobs:
        o.start()
    for o in orb_jobs:
        o.join()
    mj.join()


if __name__ == '__main__':
    main()
