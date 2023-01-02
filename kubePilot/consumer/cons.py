import pika
import os
import time
from json import dumps
import json
import socket
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
import glob
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class iterer:
    def __init__(self, x, y) -> None:
         self.x = x
         self.y = y
         self.counter = 0
         
    def __next__(self):
        start = 3*64*self.counter #chunk size 
        stop = 3*64*(self.counter+1)
        if start > self.x.shape[0] or stop > self.x.shape[0]:
            self.counter = 0
            start = 3*64*self.counter #chunk size 
            stop = 3*64*(self.counter+1)
        self.counter += 1
        return (self.x[start:stop, :,:], self.y[start:stop, :])

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

hst = os.getenv("RABBIT_HOST")
prt = os.getenv("RABBIT_PORT")

un = os.getenv("RABBIT_USERNAME")
pw = os.getenv("RABBIT_PASSWORD")
exp_id = os.getenv("EXPID")
base_dir = os.path.join("data","25")
if exp_id:
    base_dir = os.path.join("data","55")

loaded_model = tf.keras.models.load_model(os.path.join(base_dir, 'base.h5'))
json_model = loaded_model.to_json()
cfg_file = open(os.path.join(base_dir, 'config.json'), 'r')
datasets = [ g for g in glob.glob(base_dir +"/train/*X.npy")]


credentials = pika.PlainCredentials(un, pw)
parameters = pika.ConnectionParameters(hst, prt, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

os.system("rabbitmqadmin -f tsv -q list queues name | while read queue; do rabbitmqadmin -q delete queue name=${queue}; done")
channel.queue_declare(queue='aggie')
time.sleep(10)#Ensure enough time to spin-up producers
participants  = set()
for method_frame, properties, body in channel.consume('aggie'):

    data = json.loads(body)
    channel.queue_declare(queue=data['ip'])
    participants.add(data['ip'])

    # Acknowledge the message
    channel.basic_ack(method_frame.delivery_tag)

#Broadcast base model
for p in participants:
        channel.basic_publish(exchange='',
                            routing_key=p,
                            body=json_model)
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
    
channel.close()
connection.close()