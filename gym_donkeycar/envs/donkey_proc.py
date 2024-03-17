"""
file: donkey_proc.py
author: Felix Yu
date: 2018-09-12
"""
import os
import subprocess


class DonkeyUnityProcess:
    def __init__(self):
        self.proc1 = None

    # ------ Launch Unity Env ----------- #

    def start(self, sim_path: str, host: str = "0.0.0.0", port: int = 9091):
        if sim_path == "remote":
            return

        if not os.path.exists(sim_path):
            print(sim_path, "does not exist. you must start sim manually.")
            return

        port_args = ["--port", str(port), "--host", str(host), "-logFile", "unitylog.txt"]

        # Launch Unity environment
        self.proc1 = subprocess.Popen([sim_path] + port_args)

        print("donkey subprocess started")

    def quit(self) -> None:
        """
        Shutdown unity environment
        """
        if self.proc1 is not None:
            print("closing donkey sim subprocess")
            self.proc1.kill()
            self.proc1 = None
