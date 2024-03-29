import pika
import os
import logging
from pika.exchange_type import ExchangeType
import functools
import time
from rule_engines import Base_Engine, GT_fedAvg_Engine
import sys
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


class Msg:
    def __init__(self, to, frm, data, ct=None):
        self.to = to
        self.frm = frm
        self.data = data
        self.content_type = ct


class Burst_connection(ABC):
    EXCHANGE = 'federation'
    EXCHANGE_TYPE = ExchangeType.topic
    QUEUE = 'comms'

    def __init__(self, writeto:str, consumefrom: str):
        self.QUEUE = consumefrom
        self.EXCHANGE = consumefrom
        self.rk = "generic"
        self.wt = writeto
        self.msg_q = []
        self.job_q = []
        self.train_steps = 0
        self.buffer = {}
        self.max_buffer_items = 10
        self.comm_metrics  = {}
        self.msg_id = 0
        self.idata = None
        self.ping = time.time()
        self.pong = time.time()
        self.size_buffer = []
        self.comms_enabled = True
        
        hst = os.getenv("RABBIT_HOST")
        prt = os.getenv("RABBIT_PORT")

        un = os.getenv("RABBIT_USERNAME")
        pw = os.getenv("RABBIT_PASSWORD")

        credentials = pika.PlainCredentials(un, pw)
        self.parameters = pika.ConnectionParameters(hst, prt, '/', credentials, heartbeat=600,
                                                    blocked_connection_timeout=30)
        connection = pika.BlockingConnection(self.parameters)
        self._channel = connection.channel()
        self._channel.exchange_declare(exchange=self.EXCHANGE,
                                       exchange_type=self.EXCHANGE_TYPE)
        self._channel.queue_declare(queue=self.QUEUE)
        self._channel.queue_bind(self.QUEUE,
                                 self.EXCHANGE,
                                 routing_key=self.rk)

        self._channel.close()
        connection.close()
        LOGGER.warning("Init completed")
        self.re = Base_Engine(self)


    def add_msg_to_q(self, to, frm, data, ct="info"):
        msg = Msg(to, frm, data, ct)
        self.msg_q.append(msg)
        LOGGER.info("message added")

    def process_jobs(self):
        while self.job_q:
            try:
                self.re.exec(*self.job_q.pop(0))
            except Exception as e:
                print(f"Unrecognized Job attempted, trying null value error... {str(e)}")

    def publish_queue(self):
        if self.msg_q:
            connection = pika.BlockingConnection(self.parameters)
            self._channel = connection.channel()
            while self.msg_q:
                msg = self.msg_q.pop(0)
                if sys.getsizeof(msg.data):
                    self.size_buffer.append(sys.getsizeof(msg.data))
                self._channel.exchange_declare(exchange=msg.to,
                                            exchange_type=self.EXCHANGE_TYPE)
                self._channel.queue_declare(queue=msg.to)
                self._channel.queue_bind(msg.to,
                                        msg.to,
                                        routing_key=self.rk)
                properties = pika.BasicProperties(app_id='Galactic Federation',
                                                  content_type=msg.content_type,
                                                  headers={'id': self.msg_id,
                                                           'src': self.QUEUE})
                LOGGER.warning("published: " +msg.data[0:100] + " "+msg.content_type)
                self._channel.basic_publish(msg.to, 'generic', msg.data, properties)
                self.msg_id += 1
                LOGGER.info("Message sent")
                self.pong = time.time()

            self._channel.close()
            connection.close()
    def on_message_cb(self, channel, method_frame, header_frame, body):
        src = header_frame.headers['src']
        id = header_frame.headers['id']
        if src not in self.comm_metrics:
            self.comm_metrics[src] = -1
        if self.comm_metrics[src] < id:
            if sys.getsizeof(body):
                self.size_buffer.append(sys.getsizeof(body))
            self.job_q.append((method_frame, header_frame, body))
            self.comm_metrics[src] = id
        self._channel.basic_ack(method_frame.delivery_tag)
        self.ping = time.time()
            
                
    def get_messages(self):
        connection = pika.BlockingConnection(self.parameters)
        self._channel = connection.channel()
        self._channel.queue_declare(queue=self.QUEUE)
        self._channel.queue_bind(self.QUEUE,
                                 self.EXCHANGE,
                                 routing_key=self.rk)
        self._channel.basic_consume(self.QUEUE, self.on_message_cb)
        connection.process_data_events(time_limit=10)
        self._channel.close()
        connection.close()

    def process_metrics(self):
        pass

    def run(self):
        while self.comms_enabled:
            self.publish_queue()
            self.get_messages()
            self.process_jobs()
            self.process_metrics()
            time.sleep(2)

