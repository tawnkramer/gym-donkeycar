"""
IMesgHandler
Author: Tawn Kramer

Base class for a handler expected by SimClient
"""
from typing import Any, Dict

from gym_donkeycar.core.client import SDClient


class IMesgHandler(object):
    def on_connect(self, client: SDClient) -> None:
        pass

    def on_recv_message(self, message: Dict[str, Any]) -> None:
        pass

    def on_close(self) -> None:
        pass

    def on_disconnect(self) -> None:
        pass
