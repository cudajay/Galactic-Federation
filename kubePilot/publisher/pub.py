import sys
sys.path.append('/app/src/shared/')
import os
import pdb
import socket
from burst_connection import Msg, Burst_connection
from random import randint
import time

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

class Pub_BC(Burst_connection):
    def __init__(self, writeto: str, consumefrom: str):
        super().__init__(writeto, consumefrom)
        self.model = None
        self.cfg = None

def main():
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    IPAddr = str(IPAddr) + str(random_with_N_digits(5))
    i = 0
    while True:
        id_ip = IPAddr + str(i)
        print("entering")
        aggi = 'agg' + str(i)
        dmy = Pub_BC(aggi, id_ip)
        dmy.comm_metrics[aggi] = -1
        dmy.add_msg_to_q(aggi, dmy.QUEUE,dmy.QUEUE, 'init')
        dmy.run()
        time.sleep(30)
        i += 1
    
    

if __name__ == '__main__':
    main()

