import pika
import os
import logging
from pika.exchange_type import ExchangeType
import functools
import time
from message_lexicon import msg_handler
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
        self.broadcast_to = []
        self.buffer = dict()
        hst = os.getenv("RABBIT_HOST")
        prt = os.getenv("RABBIT_PORT")

        un = os.getenv("RABBIT_USERNAME")
        pw = os.getenv("RABBIT_PASSWORD")

        credentials = pika.PlainCredentials(un, pw)
        self.parameters = pika.ConnectionParameters(hst, prt, '/', credentials, heartbeat=600,
                                                    blocked_connection_timeout=300)
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


    def add_msg_to_q(self, to, frm, data, ct="unspecified"):
        msg = Msg(to, frm, data, ct)
        self.msg_q.append(msg)
        LOGGER.info("message added")

    #override this section to modify job handler
    @abstractmethod
    def process_jobs(self):
        pass

    def publish_queue(self):
        if self.msg_q:
            connection = pika.BlockingConnection(self.parameters)
            self._channel = connection.channel()
            while self.msg_q:
                msg = self.msg_q.pop(0)
                properties = pika.BasicProperties(app_id='Galactic Federation',
                                                  content_type=msg.content_type)
                LOGGER.warning(msg.data + " "+msg.content_type)
                self._channel.basic_publish(msg.to, 'generic', msg.data, properties)
                LOGGER.info("Message sent")

            self._channel.close()
            connection.close()

    def get_messages(self):
        connection = pika.BlockingConnection(self.parameters)
        self._channel = connection.channel()
        method_frame, header_frame, body = self._channel.basic_get(self.QUEUE)
        action = msg_handler(method_frame, header_frame, body)
        if action:
            self.job_q.append(action)
            self._channel.basic_ack(method_frame.delivery_tag)

        self._channel.close()
        connection.close()

    def run(self):
        while True:
            self.publish_queue()
            self.get_messages()
            self.process_jobs()
            time.sleep(5)
