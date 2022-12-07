import yaml
import multiprocessing as mp
import time
import random
from tensorflow import keras
import numpy as np
from Orbiters import LocalOrbiter


def magistrate(pipes, cfg):
    # Broadcast base model to all K
    for p in pipes:
        #randomly select from 1-3 for demo
        p.send({f'type': 'update_model', 'content': f'../models/A-{random.choice([1,2,3])}.h5'})
        p.send({'type': 'update_data', 'content': '../'})
    while pipes:
        for p in pipes:
            if p.poll():
                msg = p.recv()
                print(msg['content'])
                if msg['type'] == 'close_pipe':
                    pipes.remove(p)


def main():
    pipes = []
    orbiters = []
    orb_jobs = []
    for i in range(5):
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
