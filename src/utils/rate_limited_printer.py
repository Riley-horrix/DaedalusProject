import time

class RateLimitedPrinter:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.last_print_time = 0

    def __call__(self, *args, **kwargs):
        current_time = time.time()
        if current_time - self.last_print_time >= self.interval:
            print(*args, **kwargs)
            self.last_print_time = current_time