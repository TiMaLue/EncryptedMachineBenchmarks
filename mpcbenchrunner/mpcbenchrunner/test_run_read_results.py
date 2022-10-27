import json

import cattrs

from mpcbenchrunner.bench import Measurements

if __name__ == "__main__":
    from mpcbenchrunner import runner
    packet_stats = runner.read_network_usage_measurements(open("/tmp/tmpph6z9jlf", "rb"))

    measurements = Measurements(packet_stats=packet_stats)
    print(json.dumps(cattrs.unstructure(measurements)))