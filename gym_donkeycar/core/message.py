"""
IMesgHandler
Author: Tawn Kramer

Base class for a handler expected by SimClient
"""


class IMesgHandler(object):
    def on_connect(self, client):
        pass

    def on_recv_message(self, message):
        pass

    def on_close(self):
        pass

    def on_disconnect(self):
        pass
