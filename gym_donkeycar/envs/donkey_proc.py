'''
file: donkey_proc.py
author: Felix Yu
date: 2018-09-12
'''
import subprocess
import os


class DonkeyUnityProcess(object):

    def __init__(self):
        """
        Initialize the process.

        Args:
            self: (todo): write your description
        """
        self.proc1 = None

    ## ------ Launch Unity Env ----------- ##

    def start(self, sim_path, host='0.0.0.0', port=9091):
        """
        Starts a port

        Args:
            self: (todo): write your description
            sim_path: (str): write your description
            host: (str): write your description
            port: (int): write your description
        """

        if sim_path == "remote":
            return

        if not os.path.exists(sim_path):
            print(sim_path, "does not exist. you must start sim manually.")
            return

        port_args = ["--port", str(port),"--host", str(host), '-logFile', 'unitylog.txt']

        # Launch Unity environment
        self.proc1 = subprocess.Popen(
            [sim_path] + port_args)

        print("donkey subprocess started")

    def quit(self):
        """
        Shutdown unity environment
        """
        if self.proc1 is not None:
            print("closing donkey sim subprocess")
            self.proc1.kill()
            self.proc1 = None
