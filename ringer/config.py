import datetime
import logging

logger = logging.getLogger(__name__)

AWS_ACCESS_KEY = "your_iam_key"
AWS_SECREY_KEY = "your_secret_key"
AWS_BUCKET_NAME = "your_bucket"

HIST_POLL_TIMEOUT_SECS = 5

MAX_MOTION_CNT = 10
MAX_EVENT_LIFE = datetime.timedelta(minutes=5)

RING_USERNAME = "you@email.com"
RING_PASSWORD = "your_password"

SLACK_API_KEY = "your-api-key"
SLACK_CHANNEL = "channel-to-post-to"
SLACK_USER = "Camera Bot"
SLACK_USERNAME = "camerabot"

WORKER_SLEEP_SECS = 5

try:
    from ringer_config import *  # noqa
    import ringer_config
except ImportError:
    logger.info("Using default config")
