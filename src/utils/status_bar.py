import sys

def print_status(step: int, max_steps: int, width = 100):
    # Clear current line
    print("\x1b[G\x1b[0J", end="")
    # Print status bar
    done_pts = (step * width) // max_steps
    left_pts = width - done_pts
    status = f"{step} / {max_steps} : [" + "0" * done_pts + '-' * left_pts + "]"
    print(status, end="", flush=True)

import time

for i in range(11):
    time.sleep(0.5)
    print_status(i, 10)
print()