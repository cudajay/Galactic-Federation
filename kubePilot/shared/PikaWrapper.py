import pika
import os
from json import dumps
  
class PW:
    def __init__(self, 
                 IPAddr,
                 rhost, 
                 rport, 
                 runame,
                 rpw,
                 ) -> None:
        credentials = pika.PlainCredentials(runame, rpw)
        parameters = pika.ConnectionParameters(rhost, rport, '/', credentials)
        
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        msg = {'type': 'init', 'value': IPAddr}
        dmp = dumps(msg)
        self.ip = IPAddr
        channel.basic_publish(exchange='',
                            routing_key='aggie',
                            body=dmp)
        channel.close()
        connection.close()
        self.connection = pika.SelectConnection(parameters, on_open_callback=self.on_connected)
        
        self.channel = None
        
    def on_connected(self, connection):
        """Called when we are fully connected to RabbitMQ"""
        # Open a channel
        connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, new_channel):
        """Called when our channel has opened"""
        self.channel = new_channel
        self.channel.queue_declare(queue=self.ip, durable=True, exclusive=False, auto_delete=False, callback=self.on_queue_declared)

    def on_queue_declared(self, frame):
        """Called when RabbitMQ has told us our Queue has been declared, frame is the response from RabbitMQ"""
        self.channel.basic_consume(self.ip, self.handle_delivery, auto_ack=True)

    def handle_delivery(self, channel, method, header, body):
        """Called when we receive a message from RabbitMQ"""
        print(body)

    def run(self):
        try:
            # Loop so we can communicate with RabbitMQ
            self.connection.ioloop.start()
        except KeyboardInterrupt:
            # Gracefully close the connection
            self.connection.close()
            # Loop until we're fully closed, will stop on its own
            self.connection.ioloop.start()