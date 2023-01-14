import pika
import yaml
from shared.two_way import Dummy
from shared.burst_connection import Msg, Burst_connection
import os
import time
from json import dumps, loads
import json
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
import glob
from json import JSONEncoder
import logging
from multiprocessing import Queue

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

class Agg_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.raw_model = tf.keras.models.load_model(os.path.join('base-25.h5'))
        self.working_model = tf.keras.models.load_model(os.path.join('base-25.h5'))
        with open('config.yaml', 'r') as file:
            self.cfg = dumps(yaml.safe_load(file))

    def process_jobs(self):
        while self.job_q:
            job = self.job_q.pop(0)
            assert (type(job) is dict)
            if job['type'] == 'addQ':
                self.broadcast_to.append(job['data'])
                self.add_msg_to_q(job['data'], 
                                self.QUEUE,
                                self.raw_model.to_json(), 
                                'update_model')
                self.add_msg_to_q(job['data'], 
                                self.QUEUE,
                                self.cfg, 
                                'update_config')
                continue
            if job['type'] == 'send_model_single':
                self.add_msg_to_q(job['data'], 
                                self.QUEUE,
                                self.raw_model.to_json(), 
                                'update_model')
                continue


dmy = Agg_BC(None, 'agg')
dmy.run()

"""
base_dir = os.path.join("data","25")
if exp_id:
    base_dir = os.path.join("data","55")

loaded_model = tf.keras.models.load_model(os.path.join(base_dir, 'base.h5'))
json_model = loaded_model.to_json()
cfg_file = open(os.path.join(base_dir, 'config.json'), 'r')
datasets = [ g for g in glob.glob(base_dir +"/train/*X.npy")]

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue='aggie')
time.sleep(3)#Ensure enough time to spin-up producers
participants  = []
for method_frame, properties, body in channel.consume('aggie'):

    data = json.loads(body)
    participants.append(data['value'])

    # Acknowledge the message
    channel.basic_ack(method_frame.delivery_tag)
channel.close()
print(participants)
while True:
#Broadcast base model
    for p in participants:
            channel = connection.channel()
            #channel.queue_declare(queue=p, durable=True, exclusive=False, auto_delete=False)
            channel.basic_publish(exchange='',
                                routing_key=p,
                                body="greetings from the host")
            channel.close()


#Broadcast base config
for p in participants:
        channel.basic_publish(exchange='',
                            routing_key=p,
                            body=cfg_file.read())




# Cancel the consumer and return any pending messages
requeued_messages = channel.cancel()

# Close the channel and the connection
working_datasets = []
for i in range(3):
    sel = random.choice(datasets)
    x = np.load(sel)
    sel = sel.replace("X", "y")
    y = np.load(sel)
    working_datasets.append(iterer(x,y))

for i in range(5):
    for obj, p in zip(working_datasets, participants):
        x,y = next(obj)
        numpyData = {"x": x, "y":y}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
        channel.basic_publish(exchange='',
                    routing_key=p,
                    body=encodedNumpyData)
"""
