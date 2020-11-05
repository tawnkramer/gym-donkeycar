import time


class FPSTimer(object):
    '''
    Every N on_frame events, give the average iterations per interval.
    '''
    def __init__(self, N=100):
        """
        Initialize n times.

        Args:
            self: (todo): write your description
            N: (int): write your description
        """
        self.t = time.time()
        self.iter = 0
        self.N = N

    def reset(self):
        """
        Reset the progress bar.

        Args:
            self: (todo): write your description
        """
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        """
        Callback function to wait for time.

        Args:
            self: (todo): write your description
        """
        self.iter += 1
        if self.iter == self.N:
            e = time.time()
            print('fps', float(self.N) / (e - self.t))
            self.t = time.time()
            self.iter = 0
