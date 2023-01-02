import pika
import os
import time
from json import dumps, loads
import json
import socket
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

hst = os.getenv("RABBIT_HOST")
prt = os.getenv("RABBIT_PORT")

un = os.getenv("RABBIT_USERNAME")
pw = os.getenv("RABBIT_PASSWORD")

credentials = pika.PlainCredentials(un, pw)
parameters = pika.ConnectionParameters(hst, prt, '/', credentials)
connection = pika.BlockingConnection(parameters)

channel = connection.channel()
channel.queue_declare(queue='aggie')


dict = {'og': None, 'ip': IPAddr}
dmp = dumps(dict)

#Let aggie know device is participating
channel.basic_publish(exchange='',
                        routing_key='aggie',
                        body=dmp)
loaded_model = None
cfg = None
#Collect Base Model
go = True
while go:
    method_frame, header_frame, body = channel.basic_get('test')
    if method_frame:
        loaded_model = model_from_json(body)
        print(method_frame, header_frame, body)
        #channel.basic_ack(method_frame.delivery_tag)
        break
    else:
        continue
        
#Collect Base config
go = True
while go:
    for method_frame, properties, body in channel.consume(IPAddr):

        cfg = loads(body)
        # Acknowledge the message
        #channel.basic_ack(method_frame.delivery_tag)
        #go = False

channel.queue_delete(queue=IPAddr)
connection.close()
