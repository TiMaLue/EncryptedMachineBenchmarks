import os
import sys
import time

import docker
from docker.models.containers import Container
from dateutil import parser

time_out_sec = 30 * 60

def check_timeout(c: Container):
    api_client = docker.APIClient(base_url='unix://var/run/docker.sock')
    inspection = api_client.inspect_container(c.id)
    time_created_str = inspection["Created"]
    time_created = parser.parse(time_created_str)
    time_created_ts = time_created.timestamp()
    time_running_sec = time.time() - time_created_ts
    if time_running_sec > time_out_sec:
        return True

def watch_dog_kill_oldies():
    docker_client = docker.from_env()
    for c in docker_client.containers.list(filters={
        "status": "running",
    }):
        c: Container
        is_mpcbench_container: bool = False
        for t in c.image.tags:
            is_mpcbench_container = "mpcbenchtarget_crypten" in t
        if not is_mpcbench_container:
            continue
        container_timed_out = check_timeout(c)
        if not container_timed_out:
            # print(f"Container did not time out.")
            continue
        print(f"Benchmark container is running: {c.image} -- {c.name} -- {c.short_id}")
        print(f"Container timed out..")
        os.system(f"docker stop -t 0 {c.id}")
        print(f"Container stopped.")

def watch_dog_loop():
    while True:
        try:
            watch_dog_kill_oldies()
        except Exception:
            print("exception while trying to kill containers.")
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide at least 1 argument: Container runtime timeout in minutes.")
        exit(1)
    time_out_sec = int(60 * float(sys.argv[1]))
    watch_dog_loop()


