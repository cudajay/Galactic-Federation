import os
import random
import paramiko
import numpy as np
from tensorflow import keras
from multiprocessing import Pipe
from scp import SCPClient
import logging


class Orbiter:
    def __init__(self, id: int):
        self.id = id
        self.model = None
        self.cid = None
        self.X = None
        self.y = None
        self.test_x = None
        self.test_y = None
        self.last_hist = None
        self.updates = 0

    def load_model(self):
        pass

    def load_data(self):
        pass

    def train_model(self):
        pass

    def run(self):
        pass


class LocalOrbiter(Orbiter):
    def load_model(self, pth: os.path):
        self.model = keras.models.load_model(pth)

    def train_model(self):
        self.last_hist = self.model.fit(self.X, self.y,
                                        batch_size=70,
                                        epochs=1,
                                        validation_split=.2,
                                        verbose=True)
        self.model.save(f"{self.id}.h5")

    def load_data(self, pth: os.path):
        # NOTE: randomly select from 1-3 for demo
        num = random.choice([1, 2, 3])
        f = f'A-{num}_X.npy'
        self.X = np.load(os.path.join(pth, f))
        self.test_x = np.load(os.path.join(pth, 'test', f))
        self.test_x = self.test_x[0:100]
        f = f'A-{num}_y.npy'
        self.y = np.load(os.path.join(pth, f))
        self.test_y = np.load(os.path.join(pth, 'test', f))
        self.test_y = self.test_y[0:100]

    def get_preds(self):
        preds = self.model.predict(self.test_x)
        return f'{self.id} {self.updates} {np.mean(keras.metrics.mae(self.test_y, preds))}'

    def orbit(self, pipe: Pipe, cfg):
        while True:
            if pipe.poll():
                msg = pipe.recv()
                if msg['type'] == 'update_model':
                    self.load_model(msg['content'])
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully loaded'})
                    self.updates += 1
                    if self.updates >= 2:
                        pipe.send({'type': 'preds', 'content': self.get_preds()})
                        if self.updates >= 5:
                            pipe.send({'type': 'close_pipe', 'content': ''})
                            pipe.close()
                            break
                if msg['type'] == 'update_data':
                    self.load_data(msg['content'])
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Data successfully loaded'})
                if msg['type'] == 'train_model':
                    self.train_model()
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully trained '})
                    pipe.send({'type': 'weights', 'content': self.model})


class RemoteOrbiter(LocalOrbiter):
    def __init__(self, params: dict, id: int):
        super().__init__(id)
        self.root_dir = self.params['base_dr']
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(params['base_ip'] + params['suffix'], username=params['UserName'],
                         password=params['Password'], timeout=5)
        stdin, stdout, stderr = self.ssh.exec_command('ip addr')
        self.scp = SCPClient(self.ssh.get_transport())

    def load_data(self, pth: os.path):
        # NOTE: randomly select from 1-3 for demo
        num = random.choice([1, 2, 3])
        f = f'A-{num}_X.npy'
        self.scp.put(os.path.join(pth, f), os.path.join(self.root_dir, 'data', 'X_train.npy'))
        self.scp.put(os.path.join(pth, 'test', f), os.path.join(self.root_dir, 'data', 'X_train.npy'))

        f = f'A-{num}_y.npy'
        self.scp.put(os.path.join(pth, f), os.path.join(self.root_dir, 'data', 'y_test.npy'))
        self.scp.put(os.path.join(pth, 'test', f), os.path.join(self.root_dir, 'data', 'y_test.npy'))

    def load_model(self, pth: os.path):
        self.model = keras.models.load_model(pth)
        self.scp.put(pth, os.path.join(self.root_dir, 'model', 'working.h5'))

    def train_model(self):
        stdin, stdout, stderr = self.ssh.exec_command('python3 model_utils.py -t')
        f = f"{self.id}.h5"
        self.scp.get("working.h5", f)
        self.model = keras.models.load_model(f)

    def get_preds(self):
        stdin, stdout, stderr = self.ssh.exec_command('python3 model_utils.py -p')
        return f'{self.id} {self.updates} {stdout}'
