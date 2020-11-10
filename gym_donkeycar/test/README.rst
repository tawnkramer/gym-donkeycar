======================================================
Unit test
======================================================

Part of source code (JSON extraction from TCP flow)) is tricky and could be subject to regression.
This directory include a test program (client.test.py) that focus on testing this part.
It creates a server, wait for client connection then send various payload and verify if client succeed to extract JSON messages

To launch the test :

* install the client to be used to test (from project root directory for example):

.. code-block:: shell-session

    % pip3 install -e .

* launch the test (from this directory):

.. code-block:: shell-session

    % python3 client.test.py

If everything is OK, Result should be like :

.. code-block:: shell-session

    2020-11-10 21:02:24,165 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:24,167 - root - INFO - Incoming connection from ('127.0.0.1', 51598)
    2020-11-10 21:02:25,178 - root - INFO - Got {'msg_type': 'test3'}
    2020-11-10 21:02:25,179 - root - INFO - Got {'msg_type': 'test31'}
    .2020-11-10 21:02:26,175 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:26,186 - root - INFO - Incoming connection from ('127.0.0.1', 51600)
    2020-11-10 21:02:28,181 - root - INFO - Got {'msg_type': 'test5'}
    2020-11-10 21:02:28,182 - root - INFO - Got {'msg_type': 'test51'}
    .2020-11-10 21:02:29,183 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:29,193 - root - INFO - Incoming connection from ('127.0.0.1', 51602)
    2020-11-10 21:02:31,195 - root - INFO - Got {'msg_type': 'test6'}
    2020-11-10 21:02:31,196 - root - INFO - Got {'msg_type': 'test61'}
    .2020-11-10 21:02:32,189 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:32,200 - root - INFO - Incoming connection from ('127.0.0.1', 51604)
    2020-11-10 21:02:34,196 - root - INFO - Got {'msg_type': 'test7'}
    2020-11-10 21:02:34,196 - root - INFO - Got {'msg_type': 'test71'}
    2020-11-10 21:02:35,199 - root - INFO - Got {'msg_type': 'test72'}
    .2020-11-10 21:02:36,202 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:36,212 - root - INFO - Incoming connection from ('127.0.0.1', 51606)
    2020-11-10 21:02:38,216 - root - INFO - Got {'msg_type': 'test8'}
    2020-11-10 21:02:39,217 - root - INFO - Got {'msg_type': 'test81'}
    .2020-11-10 21:02:40,217 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:40,228 - root - INFO - Incoming connection from ('127.0.0.1', 51608)
    2020-11-10 21:02:42,222 - root - INFO - Got {'msg_type': 'test9'}
    2020-11-10 21:02:43,230 - root - INFO - Got {'msg_type': 'test91'}
    .2020-11-10 21:02:44,229 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:44,240 - root - INFO - Incoming connection from ('127.0.0.1', 51610)
    2020-11-10 21:02:45,241 - root - INFO - Got {'msg_type': 'test1'}
    .2020-11-10 21:02:46,237 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:46,248 - root - INFO - Incoming connection from ('127.0.0.1', 51612)
    2020-11-10 21:02:47,249 - root - INFO - Got {'msg_type': 'test2'}
    .2020-11-10 21:02:48,240 - gym_donkeycar.core.client - INFO - connecting to localhost:10000 
    2020-11-10 21:02:48,251 - root - INFO - Incoming connection from ('127.0.0.1', 51614)
    .2020-11-10 21:02:50,251 - root - INFO - Stoping Server
    2020-11-10 21:02:50,262 - root - INFO - Server stoped

    ----------------------------------------------------------------------
    Ran 9 tests in 27.100s

    OK
