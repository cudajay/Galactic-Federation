import os
import random
import paramiko
import numpy as np
from tensorflow import keras
from multiprocessing import Pipe
from scp import SCPClient
import logging


def wrapper_func(func, *args):
    return func(*args)


class Orbiter:
    def __init__(self, id: int, params: dict = None):
        self.id = id
        self.model = None
        self.cid = None
        self.X = None
        self.y = None
        self.test_x = None
        self.test_y = None
        self.last_hist = None
        self.updates = 0
        self.params = params

    def load_model(self):
        pass

    def load_data(self):
        pass

    def train_model(self):
        pass

    def orbit(self):
        pass

    def shutdown(self):
        pass

    def get_preds(self):
        pass

    def orbit(self, pipe: Pipe):
        if self.params:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.params['addr'], username=self.params['UserName'],
                        password=self.params['Password'], timeout=5)
            stdin, stdout, stderr = ssh.exec_command('mkdir working ; cd working')
            scp = SCPClient(ssh.get_transport())
        else:
            ssh = None
            scp = None
        while True:
            if pipe.poll():
                msg = pipe.recv()
                if msg['type'] == 'update_model':
                    self.load_model(msg['content'], ssh, scp)
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully loaded'})
                    self.updates += 1
                    if self.updates >= 2:
                        pipe.send({'type': 'preds', 'content': self.get_preds(ssh, scp)})
                        if self.updates >= 5:
                            pipe.send({'type': 'close_pipe', 'content': ''})
                            pipe.close()
                            break
                if msg['type'] == 'update_data':
                    self.load_data(msg['content'], ssh, scp)
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Data successfully loaded'})
                if msg['type'] == 'train_model':
                    self.train_model(ssh, scp)
                    pipe.send({'type': 'msg', 'content': f'{self.id}: Model successfully trained '})
                    pipe.send({'type': 'weights', 'content': self.model})
        self.shutdown(ssh, scp)


class LocalOrbiter(Orbiter):

    def load_model(self, pth: os.path, ssh, scp):
        self.model = keras.models.load_model(pth)

    def train_model(self, ssh, scp):
        self.last_hist = self.model.fit(self.X, self.y,
                                        batch_size=70,
                                        epochs=1,
                                        validation_split=.2,
                                        verbose=True)
        self.model.save(f"{self.id}.h5")

    def load_data(self, pth: os.path, ssh, scp):
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

    def get_preds(self, ssh, scp):
        preds = self.model.predict(self.test_x)
        return f'{self.id} {self.updates} {np.mean(keras.metrics.mae(self.test_y, preds))}'

    def shutdown(self, ssh, scp):
        pass


class RemoteOrbiter(LocalOrbiter):

    def load_data(self, pth: os.path, ssh, scp):
        # NOTE: randomly select from 1-3 for demo
        num = random.choice([1, 2, 3])
        f = f'A-{num}_X.npy'
        scp.put(os.path.join(pth, f), os.path.join(self.params["root_dir"], 'X_train.npy'))
        scp.put(os.path.join(pth, 'test', f), os.path.join(self.params["root_dir"], 'X_train.npy'))

        f = f'A-{num}_y.npy'
        scp.put(os.path.join(pth, f), os.path.join(self.params["root_dir"],'y_test.npy'))
        scp.put(os.path.join(pth, 'test', f), os.path.join(self.params["root_dir"], 'y_test.npy'))

    def load_model(self, pth: os.path, ssh, scp):
        self.model = keras.models.load_model(pth)
        scp.put(pth, os.path.join(self.params["root_dir"], 'working.h5'))

    def train_model(self, ssh, scp):
        stdin, stdout, stderr = ssh.exec_command('python3 model_utils.py -t')
        f = f"{self.id}.h5"
        scp.get("working.h5", f)
        self.model = keras.models.load_model(f)

    def get_preds(self, ssh, scp):
        stdin, stdout, stderr = ssh.exec_command('python3 model_utils.py -p')
        return f'{self.id} {self.updates} {stdout}'

    def shutdown(self, ssh, scp):
        stdin, stdout, stderr = ssh.exec_command('cd ../; rm -r working ')

    def run(self):
        print("meow")
