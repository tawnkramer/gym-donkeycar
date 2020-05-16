'''
author: Tawn Kramer
date: 9 Dec 2019
file: sim_client.py
notes: wraps a tcp socket client with a handler to talk to the unity donkey simulator
'''
import json
from .client import SDClient
from .message import IMesgHandler
import time

class SimClient(SDClient):
    """
      Handles messages from a single TCP client.
    """

    def __init__(self, address, msg_handler):
        # we expect an IMesgHandler derived handler
        # assert issubclass(msg_handler, IMesgHandler)
                
        # hold onto the handler
        self.msg_handler = msg_handler

        # connect to sim
        super().__init__(*address)

        # we connect right away
        msg_handler.on_connect(self)

    def send_now(self, msg):
        # takes a dict input msg, converts to json string
        # and sends immediately. right now, no queue.
        json_msg = json.dumps(msg)
        super().send_now(json_msg)

    def queue_message(self, msg):
        # takes a dict input msg, converts to json string
        # and adds to a lossy queue that sends only the last msg
        json_msg = json.dumps(msg)
        self.send(json_msg)

    def on_msg_recv(self, jsonObj):
        # pass message on to handler
        self.msg_handler.on_recv_message(jsonObj)

    def is_connected(self):
        return not self.aborted

    def __del__(self):
        self.close()

    def close(self):
        # Called to close client connection
        self.stop()

        if self.msg_handler:
            self.msg_handler.on_close()

