import os
import random
import paramiko
import numpy as np
from tensorflow import keras
from multiprocessing import Pipe
from scp import SCPClient


class Orbiter:
    def __init__(self, id: int):
        self.id = id
        self.model = None
        self.cid = None
        self.X = None
        self.y = None
        self.last_hist = None

    def load_model(self):
        pass

    def load_data(self):
        pass

    def train_model(self):
        pass

    def run(self):
        pass


class LocalOrbiter(Orbiter):
    def __init__(self, id: int):
        super.__init__(id)

    def load_model(self, pth: os.path):
        self.model = keras.models.load_model(pth)

    def train_model(self):
        self.last_hist = self.model.fit(self.X, self.y,
                                        batch_size=70,
                                        epochs=1,
                                        validation_split=.2,
                                        verbose=True)

    def load_data(self, pth: os.path):
        self.X = np.load(os.path.join(pth, self.cid + "_train_x.npy"))
        self.y = np.load(os.path.join(pth, self.cid + "_train_y.npy"))

    def run(self, pipe: Pipe):
        while True:
            if pipe.poll():
                msg = pipe.recv()
                if msg['type'] == 'update_model':
                    self.load_model(msg['content'])
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully loaded'})
                if msg['type'] == 'update_data':
                    self.load_data(msg['content'])
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Data successfully loaded'})
                    self.train_model()
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully trained '})
                    pipe.send({'type': 'msg', 'content': f'{self.id}: ' + str(self.last_hist.history['val_loss'])})
                    pipe.send({'type': 'close_pipe', 'content': ''})
                    pipe.close()
                    break


class RemoteOrbiter(LocalOrbiter):
    def __init__(self, params: dict, id: int):
        super.__init__(id)
        self.root_dir = self.params['base_dr']
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(params['base_ip'] + params['suffix'], username=params['UserName'],
                         password=params['Password'], timeout=5)
        stdin, stdout, stderr = self.ssh.exec_command('ip addr')
        self.scp = SCPClient(self.ssh.get_transport())

    def load_model(self, pth: os.path):
        self.model = keras.models.load_model(pth)
        self.scp.put(pth, os.path.join(self.root_dir, 'model', 'working.h5'))

    def load_data(self, pth: os.path):
        self.scp.put(pth, os.path.join(self.root_dir, 'data', 'X.npy'))
        self.scp.put(pth, os.path.join(self.root_dir, 'data', 'y.npy'))

    def train_model(self):
        stdin, stdout, stderr = self.ssh.exec_command('python3 remote_train.py')
        self.scp.get('working.h5', f'{self.cid}.h5')
        self.model = keras.models.load_model(f'{self.cid}.h5')

