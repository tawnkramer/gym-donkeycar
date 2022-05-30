import time


class FPSTimer:
    """
    Every N on_frame events, give the average iterations per interval.
    """

    def __init__(self, N: int = 100):
        self.last_time = time.time()
        self.iter = 0
        self.N = N

    def reset(self) -> None:
        self.last_time = time.time()
        self.iter = 0

    def on_frame(self) -> None:
        self.iter += 1
        if self.iter == self.N:
            current_time = time.time()
            print(f"fps {float(self.N) / (current_time - self.last_time):.2f}")
            self.last_time = time.time()
            self.iter = 0
