import logging

logger = logging.getLogger(__name__)

RING_USERNAME = "you@email.com"
RING_PASSWORD = "your_password"

try:
    from ringer_config import *  # noqa
    import ringer_config
except ImportError:
    logger.info("Using default config")
