import logging
from typing import ContextManager

from .env_loader import __RequiredSetting as RequiredSetting
from .env_loader import __load_from_envs as load_from_envs
from .timer import StopTimer, SimpleTimer, StopTimeAndLog


def configure_logging(log_conf="logconfig.yaml"):
    """
    Configures the logging library given the log configuration file.
    """
    if isinstance(log_conf, dict):
        import logging.config
        logging.config.dictConfig(log_conf)
    elif isinstance(log_conf, str):
        if log_conf.endswith(".yaml"):
            import yaml
            with open(log_conf, 'r') as log_fp:
                configure_logging(yaml.safe_load(log_fp))
        else:
            import logging.config
            logging.config.fileConfig(log_conf, disable_existing_loggers=False)
    else:
        raise ValueError("Configuration unrecognized: " + str(log_conf))


del logging
del ContextManager
