import pyrules
import logging
#from burst_connection import Burst_connection
from utils import IIterable
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
mse = MeanSquaredError()

from json import loads,dumps
logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

class Base_Engine:
    def __init__(self, connection_handler):
        self.ch = connection_handler
        self.rule_and_action = rules(self.handler, self.action_executor)
    def exec(self, method_frame, header_frame, body):
        self.rule_and_action(method_frame, header_frame, body)
    def handler(self, method_frame, header_frame, body):
        if method_frame:
            #1a. rec init from worker nodes
            if header_frame.content_type == 'init':
                LOGGER.info("Starting init job")
                return (self.init_job, body)
            #2a. worker, rec new model and creates jobs to set it
            if header_frame.content_type == 'update_model_snd':
                LOGGER.info("updating model")
                return (self.update_model_job, body)
            #3a. agg, request for new config from worker and creates job to send it
            if header_frame.content_type == 'update_config_req':
                LOGGER.info("send update config")
                return (self.send_update_cfg_job, body)
            #4a. worker, rec new config and creates job to set it
            if header_frame.content_type == 'update_config':
                LOGGER.info("updating config")
                return (self.update_cfg_job, body)
            #5a. agg, recs data request and creates job to send it 
            if header_frame.content_type == 'update_data_req':
                LOGGER.info(f"{body} requesting data")
                return (self.send_data_job, body)
            #6a. worker, recs data and creates job to set it
            if header_frame.content_type == 'init_data':
                LOGGER.warning(f"new data established: ")
                return (self.init_data_job, body)
            #7a. agg, recevies request to train from worker, creates job to send ok
            if header_frame.content_type == 'train_req':
                LOGGER.info(f"experiment ready {body}")
                return (self.train_ok_send, body)
            #8a, worker recevies ok to train and creates to job to start
            if header_frame.content_type == 'train_ok':
                LOGGER.info(f"experiment ready {body}")
                return (self.start_train, body)
            #9a agg, receives train results, and creates job to process them
            if header_frame.content_type == 'train_metrics':
                LOGGER.info(f"results ready ")
                return (self.process_metrics_job, body)
            #10a worker, receives new weights and creates job to process them
            if header_frame.content_type == 'update_weights':
                LOGGER.info(f"new weights added ")
                return (self.update_weights_job, data)
        else:
            return (self.null_job, None)
    def action_executor(self, job, data):
        job(data)
    def null_job(self, data):
        pass
    def init_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE,self.ch.raw_model.to_json(), 'update_model_snd')
    def update_model_job(self, data):
        self.ch.model = model_from_json(data)
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, self.ch.QUEUE,'update_config_req')
    def send_update_cfg_job(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, dumps(self.ch.cfg),'update_config')
    def update_cfg_job(self, data):
        self.ch.cfg = loads(data)
        self.ch.model.compile(loss=self.ch.cfg['loss_metric'], optimizer=self.ch.cfg['optimizer'])
        self.ch.add_msg_to_q('agg', self.ch.QUEUE,self.ch.QUEUE, "update_data_req")
    def send_data_job(self, data):
        self.ch.add_msg_to_q(data, self.QUEUE, self.ch.data_files.pop(0),'init_data')
    def init_data_job(self, data):
        LOGGER.info(data)
        x = np.load(data)
        y = np.load(data.decode('UTF-8').replace("_X", "_y"))
        self.ch.idata = IIterable(x, y, self.ch.cfg['chunk_size'])
        self.ch.add_msg_to_q('agg',  self.ch.QUEUE, self.ch.QUEUE,'train_req')
    def train_ok_send(self, data):
        self.ch.add_msg_to_q(data, self.ch.QUEUE, self.ch.QUEUE,'train_ok')
    def start_train(self, data):
        x,y = next(self.ch.idata)
        h = self.ch.model.fit(x,y,batch_size=64, epochs=8, validation_split=.2, verbose=True)
        msg = {'id': self.ch.QUEUE, 'loss': h.history['loss'][-1], 'val_loss': h.history['val_loss'][-1], 'step': self.ch.train_steps }
        for i in range(len(self.ch.model.trainable_variables)):
            msg[i] = self.ch.model.trainable_variables[i].numpy().tobytes().hex()
        self.ch.add_msg_to_q('agg', self.ch.QUEUE, dumps(msg),'train_metrics')
        self.ch.train_steps +=1
    def process_metrics_job(self, data):
        dct = loads(data)
        if dct['step'] not in self.ch.buffer:
            self.ch.buffer[dct['step']] = [dct]
        else:
            self.ch.buffer[dct['step']].append(dct)
        if len(self.ch.buffer[dct['step']]) == len(self.ch.comm_metrics.keys()):
            ids = list(map(lambda x: x['id'], self.ch.buffer[dct['step']]))
            ids = np.unique(ids)
            assert(len(ids) == len(self.ch.comm_metrics))
            m = self.ch.raw_model
            m.compile(loss=self.ch.cfg['loss_metric'], optimizer=self.ch.cfg['optimizer'])
            for i in range(len(m.trainable_variables)):
                update = np.zeros(m.trainable_variables[i].numpy().shape)
                vals = list(map(lambda x: x[str(i)], self.ch.buffer[dct['step']]))
                for v in vals:
                    update += np.ndarray(m.trainable_variables[i].numpy().shape, dtype=np.float32,buffer=bytes.fromhex(v))
                update = update/len(self.ch.comm_metrics)
                m.trainable_variables[i].assign(update)
            x,y =  next(self.ch.idata)
            yhat =  m.predict(x)
            score = mse(yhat, y)
            LOGGER.warning(f"TEST UPDATE: {score.numpy()}")
            #report scores
            #TODO
            weights = {}
            for i in range(len(m.trainable_variables)):
                weights[i] = m.trainable_variables[i].numpy().tobytes().hex()
            #send broadcast signal to update_weights
            q_items = list(self.ch.comm_metrics.keys())
            for prty in q_items:
                self.ch.add_msg_to_q(prty, self.ch.QUEUE, dumps(weights),'update_weights')
        else:
            self.ch.add_msg_to_q(dct['id'], self.ch.QUEUE, "standbye", 'info')
    def update_weights_job(self, data):
        dct = loads(data)
        for i in range(len(self.ch.model.trainable_variables)):
            weights = np.ndarray(self.ch.model.trainable_variables[i].numpy().shape, dtype=np.float32,buffer=bytes.fromhex(dct[str(i)]))
            self.ch.model.trainable_variables[i].assign(weights)
        self.ch.add_msg_to_q('agg',  self.ch.QUEUE, self.ch.QUEUE,'train_req')

                    


