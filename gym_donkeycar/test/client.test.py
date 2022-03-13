import asyncore
import logging
import socket
import sys
import time
import unittest
from threading import Thread

from gym_donkeycar.core.sim_client import SDClient

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

host = "localhost"
port = 10000


class EchoHandler(asyncore.dispatcher_with_send):
    def handle_read(self):
        data = self.recv(8192)
        if data:
            root.info("Server got %s" % data)
            self.send(data)


class TestServer(asyncore.dispatcher):
    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)
        self.processing_loop = True
        self.handler = None
        self.th = Thread(target=self.loop, args=())
        self.th.start()

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            root.info("Incoming connection from %s" % repr(addr))
            self.handler = EchoHandler(sock)

    def stop(self):
        root.info("Stoping Server")
        self.processing_loop = False
        self.th.join()
        root.info("Server stoped")

    def loop(self):
        while self.processing_loop:
            asyncore.loop(count=1)
            time.sleep(0.01)


class SUT(SDClient):
    def __init__(self, address):
        super().__init__(*address, poll_socket_sleep_time=0.01)
        self.receivedMsg = None
        self.receivedCount = 0

    def on_msg_recv(self, json_packet):
        root.info("Got %s" % json_packet)
        self.receivedMsg = json_packet
        self.receivedCount += 1

    def reInit(self):
        self.receivedMsg = None
        self.receivedCount = 0


class SDClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.server = TestServer(host, port)
        time.sleep(1)

    @classmethod
    def tearDownClass(self):
        self.server.stop()

    def setUp(self):
        self.SUT = SUT((host, port))
        time.sleep(1)
        self.SUT.reInit()

    def tearDown(self):
        self.SUT.stop()

    def test_simpleMessage(self):
        self.server.handler.send(b'{"msg_type":"test1"}\n')
        time.sleep(1)
        self.assertTrue(self.SUT.receivedCount == 1)

    def test_simpleMessageUndelimited(self):
        self.server.handler.send(b'{"msg_type":"test2"}')
        time.sleep(1)
        self.assertTrue(self.SUT.receivedCount == 1)

    def test_SimpleConcat(self):
        self.server.handler.send(b'{"msg_type":"test3"}\n{"msg_type":"test31"}')
        time.sleep(1)
        self.assertTrue(self.SUT.receivedCount == 2)

    def test_uncompletePayload(self):
        self.server.handler.send(b'{"msg_type":"test4","tutu":')
        time.sleep(1)
        self.assertTrue(self.SUT.receivedCount == 0)

    def test_fragmentedPayload1(self):
        self.server.handler.send(b'{"msg_type":"test5"')
        time.sleep(1)
        self.server.handler.send(b'}\n{"msg_type":"test51"}\n')
        time.sleep(1)
        self.assertEqual(self.SUT.receivedCount, 2)

    def test_fragmentedPayload2(self):
        self.server.handler.send(b'{"msg_type":')
        time.sleep(1)
        self.server.handler.send(b'"test6"}\n{"msg_type":"test61"}\n')
        time.sleep(1)
        self.assertEqual(self.SUT.receivedCount, 2)

    def test_fragmentedPayload3(self):
        self.server.handler.send(b'{"msg_type":"test7"')
        time.sleep(1)
        self.server.handler.send(b'}\n{"msg_type":"test71"}\n{"msg_type":')
        time.sleep(1)
        self.server.handler.send(b'"test72"}')
        time.sleep(1)
        self.assertEqual(self.SUT.receivedCount, 3)

    def test_fragmentedPayload4(self):
        self.server.handler.send(b'{"msg_type":"test8"')
        time.sleep(1)
        self.server.handler.send(b'}\n{"msg_type":')
        time.sleep(1)
        self.server.handler.send(b'"test81"}')
        time.sleep(1)
        self.assertEqual(self.SUT.receivedCount, 2)

    def test_fragmentedPayload5(self):
        self.server.handler.send(b'{"msg_type":"test9"')
        time.sleep(1)
        self.server.handler.send(b"}\n{")
        time.sleep(1)
        self.server.handler.send(b'"msg_type":"test91"}\n')
        time.sleep(1)
        self.assertEqual(self.SUT.receivedCount, 2)


if __name__ == "__main__":
    unittest.main()
