import pika
import os
import logging
from pika.exchange_type import ExchangeType
import functools

logging.basicConfig(level=logging.INFO)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
class Msg:
    def __init__(self, to, frm, data):
        self.to = to
        self.frm = frm
        self.data = data

def on_message(channel, method_frame, header_frame, body):
    print(method_frame.delivery_tag)
    LOGGER.info(body)
    print()
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    
class Burst_connection:
    EXCHANGE = 'federation'
    EXCHANGE_TYPE = ExchangeType.topic
    QUEUE = 'comms'
    def __init__(self,writeto, consumefrom):
        self.QUEUE = consumefrom
        self.EXCHANGE = consumefrom
        self.rk = "generic"
        self.wt = writeto
        self.msg_q = []
        self.job_q = []
        hst = os.getenv("RABBIT_HOST")
        prt = os.getenv("RABBIT_PORT")

        un = os.getenv("RABBIT_USERNAME")
        pw = os.getenv("RABBIT_PASSWORD")

        credentials = pika.PlainCredentials(un, pw)
        self.parameters = pika.ConnectionParameters(hst, prt, '/', credentials, heartbeat=600, blocked_connection_timeout=300)
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
    
    def add_msg_to_q(self, to, frm, data):
        msg = Msg(to, frm, data)
        self.msg_q.append(msg)
        LOGGER.info("message added")

    def publish_queue(self):
        if self.msg_q:
            connection = pika.BlockingConnection(self.parameters)
            self._channel = connection.channel()
            properties = pika.BasicProperties(app_id='Agg',
                                    content_type='application/json')
            while self.msg_q:
                msg = self.msg_q.pop(0)
                self._channel.basic_publish(msg.to, 'generic',msg.data,properties)
                LOGGER.info("Message sent")

            self._channel.close()
            connection.close()

    def get_messages(self):
        connection = pika.BlockingConnection(self.parameters)
        self._channel = connection.channel()
        channel.basic_consume(self.QUEUE, on_message)
        try:
            channel.start_consuming()
        except:
            channel.stop_consuming()
        """
        for method_frame, properties, body in self._channel.consume(self.QUEUE):

                # Display the message parts
                print(method_frame)
                print(properties)
                print(body)

                # Acknowledge the message
                self._channel.basic_ack(method_frame.delivery_tag)
                LOGGER.info("Message collected")
        """
        LOGGER.info("messages collection escaped")
        self._channel.close()
        connection.close()

    def run(self):
        while True:
            self.add_msg_to_q(self.wt, self.QUEUE, f"hello from {self.QUEUE}")
            self.publish_queue()
            self.get_messages()

                
