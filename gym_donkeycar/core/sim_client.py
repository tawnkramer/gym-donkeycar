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
        """
        Initialize the message handler.

        Args:
            self: (todo): write your description
            address: (str): write your description
            msg_handler: (todo): write your description
        """
        # we expect an IMesgHandler derived handler
        # assert issubclass(msg_handler, IMesgHandler)
                
        # hold onto the handler
        self.msg_handler = msg_handler

        # connect to sim
        super().__init__(*address)

        # we connect right away
        msg_handler.on_connect(self)

    def send_now(self, msg):
        """
        Send a message

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        # takes a dict input msg, converts to json string
        # and sends immediately. right now, no queue.
        json_msg = json.dumps(msg)
        super().send_now(json_msg)

    def queue_message(self, msg):
        """
        Send a message to the queue.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        # takes a dict input msg, converts to json string
        # and adds to a lossy queue that sends only the last msg
        json_msg = json.dumps(msg)
        self.send(json_msg)

    def on_msg_recv(self, jsonObj):
        """
        Invoked when a message is received.

        Args:
            self: (todo): write your description
            jsonObj: (todo): write your description
        """
        # pass message on to handler
        self.msg_handler.on_recv_message(jsonObj)

    def is_connected(self):
        """
        Return true if all connected component is connected.

        Args:
            self: (todo): write your description
        """
        return not self.aborted

    def __del__(self):
        """
        Closes the stream.

        Args:
            self: (todo): write your description
        """
        self.close()

    def close(self):
        """
        Close the connection.

        Args:
            self: (todo): write your description
        """
        # Called to close client connection
        self.stop()

        if self.msg_handler:
            self.msg_handler.on_close()

