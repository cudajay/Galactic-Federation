import logging
#from burst_connection import Burst_connection
from utils import IIterable
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

from json import loads,dumps
logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

def msg_handler(method_frame, header_frame, body):
    ret_dict = None
    if method_frame:
        #1a. rec init from worker nodes
        if header_frame.content_type == 'init':
            LOGGER.info("Starting init job")
            ret_dict = {'type': 'addQ', 'data': body}
        #2a. worker, rec new model and creates jobs to set it
        if header_frame.content_type == 'update_model_snd':
            LOGGER.info("updating model")
            ret_dict = {'type': 'update_model', 'data': body}
        #3a. agg, request for new config from worker and creates job to send it
        if header_frame.content_type == 'update_config_req':
            LOGGER.info("send update config")
            ret_dict = {'type': 'update_config_snd', 'data': body}
        #4a. worker, rec new config and creates job to set it
        if header_frame.content_type == 'update_config':
            LOGGER.info("updating config")
            ret_dict = {'type': 'update_config', 'data': body}
        #5a. agg, recs data request and creates job to send it 
        if header_frame.content_type == 'update_data_req':
            LOGGER.info(f"{body} requesting data")
            ret_dict = {'type': 'update_data_snd', 'data': body}
        #6a. worker, recs data and creates job to set it
        if header_frame.content_type == 'init_data':
            LOGGER.warning(f"new data established: ")
            ret_dict = {'type': 'init_data', 'data': body}
        #7a. agg, recevies request to train from worker, creates job to send ok
        if header_frame.content_type == 'train_req':
            LOGGER.info(f"experiment ready {body}")
            ret_dict = {'type': 'train_ok_snd', 'data': body}
        #8a, worker recevies ok to train and creates to job to start
        if header_frame.content_type == 'train_ok':
            LOGGER.info(f"experiment ready {body}")
            ret_dict = {'type': 'train_ok', 'data': body}
        #9a agg, receives train results, and creates job to process them
        if header_frame.content_type == 'train_metrics':
            LOGGER.info(f"results ready ")
            ret_dict = {'type': 'process_metrics', 'data': body}
        #10a worker, receives new weights and creates job to process them
        if header_frame.content_type == 'update_weights':
            LOGGER.info(f"new weights added ")
            ret_dict = {'type': 'update_weights', 'data': body}

    return ret_dict

def job_handler(slf):
    while slf.job_q:
        job = slf.job_q.pop(0)
        assert (type(job) is dict)
        #1b. agg process init job and adds worker to known list, then sends raw model
        if job['type'] == 'addQ':
            slf.add_msg_to_q(job['data'], slf.QUEUE,slf.raw_model.to_json(), 'update_model_snd')
        #2b. worker, processes job of setting model then responses with config request
        if job['type'] == 'update_model':
            slf.model = model_from_json(job['data'])
            slf.add_msg_to_q('agg', slf.QUEUE, slf.QUEUE,'update_config_req')
        #3b. agg processes job to send new config
        if job['type'] == "update_config_snd":
            slf.add_msg_to_q(job['data'], slf.QUEUE, slf.cfg,'update_config')
        #4b. worker processes job to add new config and responds with data req
        if job['type'] == 'update_config':
            slf.cfg = loads(job['data'])
            slf.model.compile(loss=slf.cfg['loss_metric'], optimizer=slf.cfg['optimizer'])
            slf.add_msg_to_q('agg', slf.QUEUE,slf.QUEUE, "update_data_req")
        #5b agg, process data send job
        if job['type'] == 'update_data_snd':
            slf.add_msg_to_q(job['data'], slf.QUEUE, slf.data_files.pop(0),'init_data')
        #6b. worker, processes job to set data and responds with request to train
        if job['type'] == 'init_data':
            LOGGER.warning(job['data'])
            x = np.load(job['data'])
            y = np.load(job['data'].decode('UTF-8').replace("_X", "_y"))
            slf.idata = IIterable(x, y, slf.cfg['chunk_size'])
            slf.add_msg_to_q('agg',  slf.QUEUE, slf.QUEUE,'train_req')
        #7b. agg, processes job to send ok to train
        if job['type'] == 'train_ok_snd':
            slf.add_msg_to_q(job['data'], slf.QUEUE, slf.QUEUE,'train_ok')
        #8b worker, processes training job and responds with training results
        if job['type'] == 'train_ok':
            x,y = next(slf.idata)
            h = slf.model.fit(x,y,batch_size=64, epochs=8, validation_split=.2, verbose=True)
            msg = {'id': slf.QUEUE, 'loss': h.history['loss'][-1], 'val_loss': h.history['val_loss'][-1], 'weigths': slf.model.train, 'step': slf.train_steps }
            for i in range(len(slf.model.trainable_variables)):
                msg[i] = slf.model.trainable_variables[i].numpy().tobytes()
            slf.add_msg_to_q('agg', slf.QUEUE, dumps(msg),'train_metrics')
            slf.train_steps +=1
        if job['type'] == "process_metrics":
            dct = loads(job['data'])
            if dct['step'] not in slf.buffer:
                slf.buffer[dct['step']] = {'id': dct}
            else:
                slf.buffer[dct['step']]['id'] =  dct 
            if slf.buffer[dct['step']] == len(slf.comm_metrics.keys()):
                ids = list(map(lambda x: x['id'], slf.buffer[dct['step']]))
                ids = np.unique(ids)
                assert(len(ids) == len(slf.comm_metrics))
                m = slf.raw_model
                m.compile(loss=slf.cfg['loss_metric'], optimizer=slf.cfg['optimizer'])
                for i in range(len(m.trainable_variables)):
                    update = np.zeros(m.trainable_variables[i].numpy().shape)
                    vals = list(map(lambda x: x[i], slf.buffer[dct['step']]))
                    for v in vals:
                        update += np.ndarray(m.trainable_variables[i].numpy().shape, dtype=np.float32,buffer=v)
                    update = update/len(slf.comm_metrics)
                    m.trainable_variables[i].assign(update)
                x,y =  next(slf.idata)
                yhat =  m.predict(x)
                score = MeanSquaredError(yhat, y)
                #report scores
                #TODO
                weights = {}
                for i in range(len(m.trainable_variables)):
                    weights[i] = m.trainable_variables[i].numpy().tobytes()
                #send broadcast signal to update_weights
                q_items = list(map(lambda x: list(x.keys())[0],slf.comm_metrics))
                for prty in q_items:
                   slf.add_msg_to_q(prty slf.QUEUE, dumps(weights),'update_weights')
                   
        if job['update_weights']:
            dct = loads(job['data'])
            for i in range(len(self.model.trainable_variables)):
                weights = np.ndarray(raw_model.trainable_variables[i].numpy().shape, dtype=np.float32,buffer=dct[i])
                self.model.trainable_variables[i].assign(weights)
            slf.add_msg_to_q('agg',  slf.QUEUE, slf.QUEUE,'train_req')
                    


