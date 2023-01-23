import sys
 
# importing sys.path
sys.path.append("/app/shared")
print(sys.path)
from async_cons import Reconnecting_async_consumer
from async_pubs import Async_pubs
import os
import pika
import threading
import socket
from multiprocessing import SimpleQueue

def on_message(channel, method_frame, header_frame, body):
    print(method_frame.delivery_tag)
    print(body)
    print()
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    
class Dummy:
    
    def __init__(self, msg = None):
        self.q = SimpleQueue()
        if msg:
            self.q.put(msg)
        
    def run(self, writeto, consumefrom):
        hst = os.getenv("RABBIT_HOST")
        prt = os.getenv("RABBIT_PORT")

        un = os.getenv("RABBIT_USERNAME")
        pw = os.getenv("RABBIT_PASSWORD")
        
        credentials = pika.PlainCredentials(un, pw)
        
        parameters = pika.ConnectionParameters(hst, prt, '/', credentials)
        pubs = Async_pubs(
            parameters, self.q, writeto
        )
        cons = Reconnecting_async_consumer(parameters, self.q, consumefrom)
        t1 = threading.Thread(target=cons.run)
        t2 = threading.Thread(target=pubs.run)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
