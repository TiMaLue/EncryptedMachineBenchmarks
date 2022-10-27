import time
import signal

def handle_err():
        print("Handled Error")
try:
    print("Going to sleep")
    to_print = "Sleep interrupted"

    signal.signal(signal.SIGINT, lambda _: RuntimeError())
    signal.signal(signal.SIGTERM, lambda _: RuntimeError())

    time.sleep(100)
    print("Sleep finished")
finally:
    print("finally block")
    print("Handled Error")
