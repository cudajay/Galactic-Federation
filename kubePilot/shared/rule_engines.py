import logging
from utils import IIterable, Rule
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import os
from json import loads, dumps
import random
import time
import sys, traceback
mse = MeanSquaredError()

from json import loads, dumps

logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

def padd(x:np.ndarray):
    if x.shape[-1] ==25:
        shp = (*x.shape[:2], 55)
        z = np.zeros(shp)
        z[:shp[0],:shp[1],:25] = x
        return z
    else:
        return x

def data_handler(data):
    if type(data) == bytes:
        return data.decode()
    if type(data) == str:
        try:
            return loads(data)
        except:
            print("\nproceeding with data as string\n")
    return data


class Base_Engine:
    def __init__(self, connection_handler):
        self.ch = connection_handler
        self.rule_and_action = Rule(self.handler, self.action_executor)

    def exec(self, method_frame, header_frame, body):
        self.rule_and_action.execute(method_frame, header_frame, body)

    def handler(self, method_frame, header_frame, body):
        if method_frame:
            # 1a. rec init from worker nodes
            if header_frame.content_type == 'init':
                LOGGER.info("Starting init job")
                return (self.init_job, body)
            # 2a. worker, rec new model and creates jobs to set it
            if header_frame.content_type == 'update_model_snd':
                LOGGER.info("updating model")
                return (self.update_model_job, body)
            # 3a. agg, request for new config from worker and creates job to send it
            if header_frame.content_type == 'update_config_req':
                LOGGER.info("send update config")
                return (self.send_update_cfg_job, body)
            # 4a. worker, rec new config and creates job to set it
            if header_frame.content_type == 'update_config':
                LOGGER.info("updating config")
                return (self.update_cfg_job, body)
            # 5a. agg, recs data request and creates job to send it
            if header_frame.content_type == 'update_data_req':
                LOGGER.info(f"{body} requesting data")
                return (self.send_data_job, body)
            # 6a. worker, recs data and creates job to set it
            if header_frame.content_type == 'init_data':
                LOGGER.warning(f"new data established: ")
                return (self.init_data_job, body)
            # 7a. agg, recevies request to train from worker, creates job to send ok
            if header_frame.content_type == 'train_req':
                LOGGER.info(f"experiment ready {body}")
                return (self.train_ok_send, body)
            # 8a, worker recevies ok to train and creates to job to start
            if header_frame.content_type == 'train_ok':
                LOGGER.info(f"experiment ready {body}")
                return (self.start_train, body)
            # 9a agg, receives train results, and creates job to process them
            if header_frame.content_type == 'train_metrics':
                LOGGER.info(f"results ready ")
                return (self.process_metrics_job, body)
            # 10a worker, receives new weights and creates job to process them
            if header_frame.content_type == 'update_weights':
                LOGGER.info(f"new weights added ")
                return (self.update_weights_job, body)
            if header_frame.content_type == 'reset':
                LOGGER.info(f"federation resetted ")
                return (self.reset, body)
            if header_frame.content_type == "info":
                return( self.null_job, body)
        else:
            return (self.null_job, [])

    def action_executor(self, job, data):
        try:
            job(data_handler(data))
        except Exception as e:
                LOGGER.error(e)
                LOGGER.error(self.ch)
                LOGGER.error(traceback.print_exc(file=sys.stdout))


    def null_job(self, data):
        pass

    def init_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.raw_model.to_json(), 'update_model_snd')

    def update_model_job(self, data):
        self.ch.model = model_from_json(data)
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, 'update_config_req')

    def send_update_cfg_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, dumps(self.ch.cfg), 'update_config')

    def update_cfg_job(self, data):
        self.ch.cfg = loads(data)
        self.ch.model.compile(loss=self.ch.cfg['loss_metric'], optimizer=self.ch.cfg['optimizer'])
        if self.ch.cfg['re'] == 'fedAvg':
            self.ch.re = GT_fedAvg_Engine(self.ch)
        if self.ch.cfg['re'] == 'fedSgd':
            self.ch.re = GT_fedsgd_engine(self.ch)
        if self.ch.cfg['re'] == 'fedAvg-h':
            self.ch.re = GT_fedAvgh_Engine(self.ch)
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, "update_data_req")

    def send_data_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.get_next_data(), 'init_data')

    def init_data_job(self, data):
        LOGGER.info(data)
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.ch.idata = IIterable(x, y, self.ch.cfg['chunk_size'])
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, 'train_req')

    def train_ok_send(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.QUEUE, 'train_ok')
        
    def start_train(self, data):
        x, y = next(self.ch.idata)
        x = padd(x)
        h = self.ch.model.fit(x, y, batch_size=64, epochs=8, verbose=True)
        msg = {'id': self.ch.QUEUE, 'loss': h.history['loss'][-1], 'step': self.ch.train_steps}
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = self.ch.model.trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1

    def process_metrics_job(self, data):
        dct = loads(data)
        if self.ch.buffer[-1] > dct['step']:
            return
        if dct['step'] not in self.ch.buffer:
            self.ch.buffer[dct['step']] = [dct]
        else:
            self.ch.buffer[dct['step']].append(dct)
        ids = set(map(lambda x: x['id'], self.ch.buffer[dct['step']]))
        if len(ids.intersection(self.ch.C)) == len(self.ch.C):
            data = {}
            for d in self.ch.buffer[dct['step']]:
                data[f"{d['id']}-step"] = d['step']
                data[f"{d['id']}-loss"] = float(np.ndarray(1, dtype=np.float32,buffer=bytes.fromhex(d['loss']))[0])

            for i in range(len(self.ch.raw_model.trainable_variables)):
                update = np.zeros(self.ch.raw_model.trainable_variables[i].numpy().shape)
                vals = list(map(lambda x: x[str(i)], self.ch.buffer[dct['step']]))
                for v in vals:
                    update += np.ndarray(self.ch.raw_model.trainable_variables[i].numpy().shape, dtype=np.float32,
                                         buffer=bytes.fromhex(v))
                update = update / len(self.ch.comm_metrics)
                self.ch.raw_model.trainable_variables[i].assign(update)

            scores = []
            for id in self.ch.idata:
                x, y = next(id)
                x = padd(x)
                yhat = self.ch.raw_model.predict(x)
                scores.append(mse(yhat, y).numpy())
            data['global-loss'] = float(np.mean(scores))

            if int(dct['step']) > 1:
                less_than = bool(self.ch.last_target_score > data['global-loss'])
                if less_than and self.ch.cfg['direction_min']:
                    self.ch.raw_model.save(os.path.join(self.ch.run_metrics_location, "model.h5"))
                    self.ch.g_min = data['global-loss']
                    self.ch.patience_test = 0
                else:
                    self.ch.patience_test += 1
            self.ch.run_data.append(data)
            save_file = open(os.path.join(self.ch.run_metrics_location, "training.json"), "w")
            json.dump(self.ch.run_data, save_file, indent=6)
            save_file.close()

            LOGGER.warning(f"TEST UPDATE: {data['global-loss']}")
            weights = {}
            for i in range(len(self.ch.raw_model.trainable_variables)):
                weights[i] = self.ch.raw_model.trainable_variables[i].numpy().tobytes().hex()
            # send broadcast signal to update_weights as long patience is within tolerance
            if self.ch.patience_test < self.ch.cfg['patience']:
                self.post_agg_processing(weights)
            else:
                print("***Patience exceeded, experiment terminated!***\n\n\n")
                q_items = list(self.ch.comm_metrics.keys())
                for prty in q_items:
                    self.ch.add_msg_to_q(prty, self.ch.QUEUE, "blank", 'reset')
                self.ch.comms_enabled = False
            self.ch.last_target_score = data['global-loss']
            self.ch.round += 1
            if len(self.ch.buffer) > 5:
                self.ch.buffer.pop(0)
        else:
            self.ch.add_msg_to_q(dct['id'], self.ch.QUEUE, "standbye", 'info')

    def post_agg_processing(self, weights):
        self.ch.C = set(random.choices(list(self.ch.comm_metrics.keys()),
                        k = int(self.ch.cfg['C']*len(self.ch.comm_metrics.keys()))))
        q_items = list(self.ch.comm_metrics.keys())
        for prty in q_items:
            self.ch.add_msg_to_q(prty, self.ch.QUEUE, dumps(weights), 'update_weights')
        
    def update_weights_job(self, data):
        dct = loads(data)
        for i in range(len(self.ch.model.trainable_variables)):
            weights = np.ndarray(self.ch.model.trainable_variables[i].numpy().shape, dtype=np.float32,
                                 buffer=bytes.fromhex(dct[str(i)]))
            self.ch.model.trainable_variables[i].assign(weights)
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, 'train_req')
        
    def reset(self, data):
        self.ch.comms_enabled = False
        LOGGER.warning("****\n\n\n\n node resetting \n\n\n\n*****")



class GT_fedAvg_Engine(Base_Engine):
    def __init__(self, connection_handler) -> None:
        super().__init__(connection_handler)
        self.opt = Adam(learning_rate=self.ch.cfg['lr'], decay=self.ch.cfg['lr'] / self.ch.cfg['epochs'])
    def init_data_job(self, data):
        LOGGER.warning(f"initing {data} with super")
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.ch.idata = IIterable(x, y, self.ch.cfg['chunk_size'])
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, 'train_req')
    def start_train(self, data):
        x, y = next(self.ch.idata)
        x = padd(x)
        tl = 0
        t1 = time.time()
        if self.ch.train_steps:
            LOGGER.warning("Time since last train {:.2f} seconds".format(t1 - self.last_train))
        for _ in range(self.ch.cfg['epochs']):
            tl = self.step(x, y)
            LOGGER.warning(f"loss: {tl}")
        t2 = time.time()
        self.last_train = t2
        LOGGER.warning("Elapsed time: {:.2f} seconds".format(t2-t1))
        msg = {'id': self.ch.QUEUE, 'loss': tl.numpy().tobytes().hex(),
               'step': self.ch.train_steps}
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = self.ch.model.trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1

    def step(self, X, y):
        loss = None
        with tf.GradientTape() as tape:
            pred = self.ch.model(X)
            loss = mse(y, pred)

        grads = tape.gradient(loss, self.ch.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.ch.model.trainable_variables))
        return loss
    
class GT_fedsgd_engine(GT_fedAvg_Engine):
    def __int__(self, connection_handler):
        super().__init__(connection_handler)

    def start_train(self, data):
        x, y = next(self.ch.idata)
        loss = None
        with tf.GradientTape() as tape:
            pred = self.ch.model(x)
            loss = mse(y, pred)

        grads = tape.gradient(loss, self.ch.model.trainable_variables)
        LOGGER.warning(f"loss: {loss}")
        msg = {'id': self.ch.QUEUE, 'loss': loss.numpy().tobytes().hex(),
               'step': self.ch.train_steps}
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = grads[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1

class GT_fedAvgh_Engine(GT_fedAvg_Engine):
    def __int__(self, connection_handler):
        super().__init__(connection_handler)

    def start_train(self, data):
        x, y = next(self.ch.idata)
        x = padd(x)
        tl = 0
        t1 = time.time()
        if self.ch.train_steps:
            LOGGER.warning("Time since last train {:.2f} seconds".format(t1 - self.last_train))
        for _ in range(self.ch.cfg['epochs']):
            tl = self.step(x, y)
            LOGGER.warning(f"loss: {tl}")
        t2 = time.time()
        self.last_train = t2
        LOGGER.warning("Elapsed time: {:.2f} seconds".format(t2-t1))
        msg = {'id': self.ch.QUEUE, 'loss': tl.numpy().tobytes().hex(),
               'step': self.ch.train_steps}
        for i in range(len(self.ch.model.layers[1].trainable_variables)):
            msg[i] = self.ch.model.layers[1].trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1

    def process_metrics_job(self, data):
        dct = loads(data)
        self.ch.buffer[dct['step']].append(dct)
        ids = set(map(lambda x: x['id'], self.ch.buffer[self.ch.step]))
        if (len(ids.intersection(self.ch.C)) == len(self.ch.C) and
            self.ch.round != 0) or (self.ch.round == 0 and len(ids)>= 10):
            data = {}
            for d in self.ch.buffer[self.ch.step]:
                data[f"{d['id']}-step"] = d['step']
                data[f"{d['id']}-loss"] = float(np.ndarray(1, dtype=np.float32,buffer=bytes.fromhex(d['loss']))[0])

            for i in range(len(self.ch.tail.trainable_variables)):
                update = np.zeros(self.ch.tail.trainable_variables[i].numpy().shape)
                vals = list(map(lambda x: x[str(i)], self.ch.buffer[self.ch.step]))
                for v in vals:
                    update += np.ndarray(self.ch.tail.trainable_variables[i].numpy().shape, dtype=np.float32,
                                         buffer=bytes.fromhex(v))
                update = update / len(self.ch.comm_metrics)
                self.ch.tail.trainable_variables[i].assign(update)

            scores = []
            for id in self.ch.idata:
                x, y = next(id)
                x = padd(x)
                yhat = self.ch.tail.predict(x)
                scores.append(mse(yhat, y).numpy())
            data['global-loss'] = float(np.mean(scores))

            if int(self.ch.step) > 1:
                less_than = bool(self.ch.last_target_score > data['global-loss'])
                if less_than and self.ch.cfg['direction_min']:
                    self.ch.tail.save(os.path.join(self.ch.run_metrics_location, "tail.h5"))
                    self.ch.g_min = data['global-loss']
                    self.ch.patience_test = 0
                else:
                    self.ch.patience_test += 1
            self.ch.run_data.append(data)
            save_file = open(os.path.join(self.ch.run_metrics_location, "training.json"), "w")
            json.dump(self.ch.run_data, save_file, indent=6)
            save_file.close()

            LOGGER.warning(f"TEST UPDATE: {data['global-loss']}")
            weights = {}
            for i in range(len(self.ch.tail.trainable_variables)):
                weights[i] = self.ch.tail.trainable_variables[i].numpy().tobytes().hex()
            # send broadcast signal to update_weights as long patience is within tolerance
            if self.ch.patience_test < self.ch.cfg['patience']:
                self.post_agg_processing(weights)
            else:
                print("***Patience exceeded, experiment terminated!***\n\n\n")
                q_items = list(self.ch.comm_metrics.keys())
                for prty in q_items:
                    self.ch.add_msg_to_q(prty, self.ch.QUEUE, "blank", 'reset')
                self.ch.comms_enabled = False
            self.ch.last_target_score = data['global-loss']
            self.ch.round += 1
            self.ch.step += 1
            self.ch.buffer[self.ch.step] =[]
        else:
            self.ch.add_msg_to_q(dct['id'], self.ch.QUEUE, "standbye", 'info')

    def update_weights_job(self, data):
        dct = loads(data)
        for i in range(len(self.ch.model.layers[1].trainable_variables)):
            weights = np.ndarray(self.ch.model.layers[1].trainable_variables[i].numpy().shape, dtype=np.float32,
                                 buffer=bytes.fromhex(dct[str(i)]))
            self.ch.model.layers[1].trainable_variables[i].assign(weights)
        self.ch.add_msg_to_q(self.ch.wt, self.ch.QUEUE, self.ch.QUEUE, 'train_req')
        
        
'''
# Sparse coding method
from sklearn.decomposition import DictionaryLearning

# create a dictionary learning object
dl = DictionaryLearning(n_components=50, alpha=1, max_iter=1000)

# fit the dictionary to the data
dl.fit(X)

# use the learned dictionary to encode the data
Z = dl.transform(X)

# use the learned dictionary to decode the data
X_reconstructed = dl.inverse_transform(Z)

import numpy as np

def dictionary_learning(X, n_components, n_iterations, learning_rate):
    n_samples, n_features = X.shape

    # Initialize the dictionary
    D = np.random.randn(n_features, n_components)

    # Iterate until convergence
    for i in range(n_iterations):
        # Compute the sparse codes using L1-minimization
        codes = l1_minimization(X, D)

        # Compute the dictionary update using gradient descent
        D_grad = np.dot(X.T, codes)
        D -= learning_rate * D_grad

    return D

def l1_minimization(X, D):
    # Compute the sparse codes using L1-minimization
    codes = np.zeros((D.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        codes[:, i] = l1_minimization_single(X[:, i], D)

    return codes

def l1_minimization_single(x, D):
    alpha = 0.1  # L1 regularization parameter
    max_iter = 100  # Maximum number of iterations

    # Initialize the sparse code
    z = np.zeros(D.shape[1])

    # Iterate until convergence
    for i in range(max_iter):
        # Compute the gradient
        grad = np.dot(D.T, np.dot(D, z) - x) + alpha * np.sign(z)

        # Update the sparse code
        z -= grad * alpha

        # Project onto the L1 ball
        z = project_l1_ball(z)

    return z

def project_l1_ball(x):
    # Project x onto the L1 ball
    u = np.abs(x)
    s = np.sum(u)
    if s <= 1:
        return x
    else:
        rho = 1 / np.sqrt(len(x))
        w = (s - 1) / len(x)
        return np.sign(x) * np.maximum(u - rho * (u - w), 0)

# Example usage
X = np.random.randn(100, 50)
D = dictionary_learning(X, n_components=10, n_iterations=100, learning_rate=0.01)


'''
