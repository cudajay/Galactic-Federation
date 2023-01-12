import pika
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

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class iterer:
    def __init__(self, x, y, chunk_size) -> None:
         self.x = x
         self.y = y
         self.counter = 0
         self.chunk_size = chunk_size
         
    def __next__(self):
        start = self.chunk_size*self.counter #chunk size 
        stop = self.chunk_size*(self.counter+1)
        if start > self.x.shape[0] or stop > self.x.shape[0]:
            self.counter = 0
            start = self.chunk_size*self.counter #chunk size 
            stop = self.chunk_size*(self.counter+1)
        self.counter += 1
        return (self.x[start:stop, :,:], self.y[start:stop, :])


dmy = Burst_connection('agg', 'agg')
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