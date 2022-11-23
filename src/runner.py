import yaml
import multiprocessing as mp
import time
import random
from tensorflow import keras
import numpy as np


def magistrate(pipes, cfg):
    # Broadcast base model to all K
    for p in pipes:
        p.send({'type': 'update_model', 'content': 'A-1.h5'})
        p.send({'type': 'update_data', 'content': ''})
    while pipes:
        for p in pipes:
            if p.poll():
                msg = p.recv()
                print(msg['content'])
                if msg['type'] == 'close_pipe':
                    pipes.remove(p)


def feds(pipe, cfg):
    model = []
    cont = True
    while cont:
        if pipe.poll():
            msg = pipe.recv()
            if msg['type'] == 'update_model':
                model = keras.models.load_model(msg['content'])
                pipe.send({'type': 'msg', 'content': 'Model successfully loaded'})
            if msg['type'] == 'update_data':
                x = np.load(msg['content'] + 'pilot_X.npy')
                y = np.load(msg['content'] + 'pilot_y.npy')
                pipe.send({'type': 'msg', 'content': 'data successfully loaded'})
                hist = model.fit(x, y,
                                 batch_size=70,
                                 epochs=1,
                                 validation_split=.2,
                                 verbose=True)
                pipe.send({'type': 'msg', 'content': str(hist.history['val_loss'])})
                cont = False
                pipe.send({'type': 'close_pipe', 'content': ''})
                pipe.close()
                break


def main():
    pipes = []
    orbiters = []
    for _ in range(5):
        x, y = mp.Pipe()
        pipes.append(x)
        orbiters.append(mp.Process(target=feds, args=(y, 'f',)))
    mj = mp.Process(target=magistrate, args=(pipes, 'f',))
    mj.start()
    for o in orbiters:
        o.start()
    for o in orbiters:
        o.join()
    mj.join()


if __name__ == '__main__':
    main()
