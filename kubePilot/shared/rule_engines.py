import logging
from shared.utils import IIterable, Rule
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import os
from json import loads, dumps
import random

mse = MeanSquaredError()

from json import loads, dumps

logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


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
        else:
            return (self.null_job, [])

    def action_executor(self, job, data):
        job(data_handler(data))

    def null_job(self, data):
        pass

    def init_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.raw_model.to_json(), 'update_model_snd')

    def update_model_job(self, data):
        self.ch.model = model_from_json(data)
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE, 'update_config_req')

    def send_update_cfg_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, dumps(self.ch.cfg), 'update_config')

    def update_cfg_job(self, data):
        self.ch.cfg = loads(data)
        self.ch.model.compile(loss=self.ch.cfg['loss_metric'], optimizer=self.ch.cfg['optimizer'])
        if self.ch.cfg['re'] == 'fedAvg':
            self.ch.re = GT_fedAvg_Engine(self.ch)
        if self.ch.cfg['re'] == 'fedSgd':
            self.ch.re = GT_fedsgd_engine(self.ch)
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE, "update_data_req")

    def send_data_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.data_files.pop(0), 'init_data')

    def init_data_job(self, data):
        LOGGER.info(data)
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.ch.idata = IIterable(x, y, self.ch.cfg['chunk_size'])
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE, 'train_req')

    def train_ok_send(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.QUEUE, 'train_ok')
        
    def start_train(self, data):
        x, y = next(self.ch.idata)
        h = self.ch.model.fit(x, y, batch_size=64, epochs=8, verbose=True)
        msg = {'id': self.ch.QUEUE, 'loss': h.history['loss'][-1], 'step': self.ch.train_steps}
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = self.ch.model.trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1

    def process_metrics_job(self, data):
        dct = loads(data)
        if dct['step'] not in self.ch.buffer:
            self.ch.buffer[dct['step']] = [dct]
        else:
            self.ch.buffer[dct['step']].append(dct)
        ids = set(map(lambda x: x['id'], self.ch.buffer[dct['step']]))
        if (ids.intersection(self.ch.C) == self.ch.C) or (len(ids) == len(list(self.ch.comm_metrics.keys()))):
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
            x, y = next(self.ch.idata)
            yhat = self.ch.raw_model.predict(x)
            score = mse(yhat, y)
            data['global-loss'] = float(score.numpy())
            if data['global-loss'] < self.ch.g_min:
                self.ch.raw_model.save(os.path.join(self.ch.run_metrics_location, "model.h5"))
                self.ch.g_min = data['global-loss']
                self.ch.patience_test = 0
            else:
                self.ch.patience_test += 1
            self.ch.run_data.append(data)
            save_file = open(os.path.join(self.ch.run_metrics_location, "training.json"), "w")
            json.dump(self.ch.run_data, save_file, indent=6)
            save_file.close()

            LOGGER.warning(f"TEST UPDATE: {score.numpy()}")
            weights = {}
            for i in range(len(self.ch.raw_model.trainable_variables)):
                weights[i] = self.ch.raw_model.trainable_variables[i].numpy().tobytes().hex()
            # send broadcast signal to update_weights as long patience is within tolerance
            if self.ch.patience_test < self.ch.cfg['patience']:
                self.post_agg_processing(weights)
            else:
                print("Patience exceeded, experiment terminated!*100")
                self.ch.comms_enabled = False
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
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE, 'train_req')




class GT_fedAvg_Engine(Base_Engine):
    def __init__(self, connection_handler) -> None:
        super().__init__(connection_handler)
        self.opt = Adam(learning_rate=self.ch.cfg['lr'], decay=self.ch.cfg['lr'] / self.ch.cfg['epochs'])
    def init_data_job(self, data):
        LOGGER.warning(f"initing {data} with super")
        x = np.load(data)
        y = np.load(data.replace("_X", "_y"))
        self.ch.idata = IIterable(x, y, self.ch.cfg['chunk_size'])
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE, 'train_req')
    def start_train(self, data):
        x, y = next(self.ch.idata)
        tl = 0
        for _ in range(self.ch.cfg['epochs']):
            tl = self.step(x, y)
            LOGGER.warning(f"loss: {tl}")
        msg = {'id': self.ch.QUEUE, 'loss': tl.numpy().tobytes().hex(),
               'step': self.ch.train_steps}
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = self.ch.model.trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, dumps(msg), 'train_metrics')
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
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, dumps(msg), 'train_metrics')
        self.ch.train_steps += 1
