"""
IMesgHandler
Author: Tawn Kramer

Base class for a handler expected by SimClient
"""

class IMesgHandler(object):

    def on_connect(self, client):
        """
        Called when the client.

        Args:
            self: (todo): write your description
            client: (todo): write your description
        """
        pass

    def on_recv_message(self, message):
        """
        Handle a message.

        Args:
            self: (todo): write your description
            message: (str): write your description
        """
        pass

    def on_close(self):
        """
        Closes the connection.

        Args:
            self: (todo): write your description
        """
        pass

    def on_disconnect(self):
        """
        Called when the connection.

        Args:
            self: (todo): write your description
        """
        pass

