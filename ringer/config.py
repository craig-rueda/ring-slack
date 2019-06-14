import logging

logger = logging.getLogger(__name__)

AWS_ACCESS_KEY = "your_iam_key"
AWS_SECREY_KEY = "your_secret_key"
AWS_BUCKET_NAME = "your_bucket"

HIST_POLL_TIMEOUT_SECS = 10

RING_USERNAME = "you@email.com"
RING_PASSWORD = "your_password"

SLACK_API_KEY = "your-api-key"
SLACK_CHANNEL = "channel-to-post-to"
SLACK_USER = "Camera Bot"
SLACK_USERNAME = "camerabot"

try:
    from ringer_config import *  # noqa
    import ringer_config
except ImportError:
    logger.info("Using default config")
