import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from json import dumps, loads
from pathlib import Path
from queue import Queue
from threading import Thread

import boto3
import coloredlogs
import cv2
import requests
from PIL import Image
from apng import APNG
from oauthlib.oauth2 import MissingTokenError
from ring_doorbell import Ring, Auth
from slacker import Slacker

from .config import (
    AWS_ACCESS_KEY,
    AWS_BUCKET_NAME,
    AWS_SECREY_KEY,
    HIST_POLL_TIMEOUT_SECS,
    MAX_EVENT_LIFE,
    MAX_MOTION_CNT,
    RING_PASSWORD,
    RING_USERNAME,
    SLACK_API_KEY,
    SLACK_CHANNEL,
    SLACK_USER,
    SLACK_USERNAME,
    WORKER_SLEEP_SECS
)

logger = logging.getLogger(__name__)
event_queue = Queue()
cache_file = Path("{}/.ringer_token.cache".format(Path.home().absolute()))


def token_updated(token):
    cache_file.write_text(dumps(token))


def otp_callback():
    return input("2FA code: ")


def gobble_frames(video, frames_to_gobble=5):
    check, frame = video.read()
    read_frames = 0
    while check and read_frames < frames_to_gobble:
        check, frame = video.read()
        read_frames += 1

    return read_frames


def process_motion(video, temp_dir):
    static_back = None
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    motion_count = 0

    # Gobble up the first few frames first
    frames_read = gobble_frames(video, frames_to_gobble=10)

    while motion_count < MAX_MOTION_CNT:
        # Reading frame(image) from video
        frames_remaining, frame = video.read()
        if not frames_remaining:
            logger.debug("End of input read {} of {} frames", frames_read, frame_count)
            break

        frames_read += 1

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur
        # so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # In first iteration we assign the value
        # of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Difference between static background
        # and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Finding contour of moving object
        (_, cnts, _) = cv2.findContours(thresh_frame.copy(),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_motion = False
        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue

            found_motion = True
            # Looks like we found some motion, so let's draw a box around it...
            (x, y, w, h) = cv2.boundingRect(contour)
            # making green rectangle arround the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if found_motion:
            img_location = os.path.join(temp_dir, "{0:03d}.png".format(motion_count))
            cv2.imwrite(img_location, frame)
            motion_count += 1

        frames_read += gobble_frames(video)


def fetch_video(video_url, temp_dir):
    r = requests.get(video_url)
    video_file = os.path.join(temp_dir, "src.mp4")

    with open(video_file, "wb") as f:
        f.write(r.content)

    return video_file


def handle_video(video_url, event, device):
    temp_dir = tempfile.mkdtemp()
    img_full_dir = os.path.join(temp_dir, "full")
    img_thumb_dir = os.path.join(temp_dir, "thumb")
    result_thumb_png = os.path.join(img_thumb_dir, "result.png")
    os.mkdir(img_full_dir)
    os.mkdir(img_thumb_dir)

    logger.info("Processing video for device {}...".format(device))

    try:
        video_file = fetch_video(video_url, temp_dir)
        video = cv2.VideoCapture(video_file)
    except Exception as e:
        logger.exception(e)
        return

    try:
        process_motion(video, img_full_dir)

        image_full_files = [os.path.join(img_full_dir, f) for f in
                            os.listdir(img_full_dir)]
        image_full_files.sort()
        if not image_full_files:
            logger.warning("Found no motion in video {}...".format(video_url))
            return

        # Now, resize the captured frames
        image_thumb_files = []
        for f in image_full_files:
            thumb_file = os.path.join(img_thumb_dir, os.path.basename(f))
            image = Image.open(f)
            image.thumbnail((350, 350), Image.ANTIALIAS)
            image.save(thumb_file, 'PNG')
            image_thumb_files.append(thumb_file)

        # Build some animated Gifs
        APNG.from_files(image_thumb_files, delay=500).save(result_thumb_png)

        # Upload to S3
        key = uuid.uuid4().hex
        thumb_url = upload_thumb_to_s3(result_thumb_png, "thumb", key, "image/png")
        video_url = upload_thumb_to_s3(video_file, "vid", key, "video/mp4")

        # Now, post to slack
        created_at_local = utc_to_local(event['created_at'])
        post_to_slack("Motion detected by *{}* at *{}*".format(
            device.name, created_at_local.ctime()),
                      "Full Video",
                      video_url,
                      thumb_url)

    except Exception as e:
        logger.exception(e)
    finally:
        shutil.rmtree(temp_dir)
        video.release()


def get_latest_recording(doorbell):
    history = doorbell.history(limit=1)
    last_hist = {"id": 0}
    if history:
        last_hist = history[0]

    return last_hist


def upload_thumb_to_s3(file, folder, key, content_type):
    full_key = "{}/{}".format(folder, key)
    s3_client.upload_file(
        file,
        AWS_BUCKET_NAME,
        full_key,
        ExtraArgs={
            "ContentType": content_type,
            "StorageClass": "REDUCED_REDUNDANCY",
        }
    )

    return "https://s3.amazonaws.com/{}/{}".format(AWS_BUCKET_NAME, full_key)


def configure_logger():
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = \
        dict(color="yellow", bold=coloredlogs.CAN_USE_BOLD_FONT)
    coloredlogs.install(level="INFO",
                        fmt="%(asctime)s %(name)-5s [%(levelname)-5s] %(message)s",
                        )


def post_to_slack(msg, subject, full_url, thumb_url):
    slack.chat.post_message(
        SLACK_CHANNEL,
        msg.strip(),
        as_user=SLACK_USER,
        username=SLACK_USERNAME,
        attachments=[{
            'title': subject.strip(),
            'title_link': full_url,
            'image_url': thumb_url
        }]
    )


def enqueue_event(event, device):
    event_queue.put({"device": device, "event": event, "added_at": datetime.now()})


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def worker_loop():
    while True:
        event_dict = event_queue.get()
        added_at = event_dict["added_at"]
        device = event_dict["device"]
        event = event_dict["event"]
        event_id = event["id"]

        logger.info("Handling event {}...".format(event_dict))

        if added_at + MAX_EVENT_LIFE <= datetime.now():
            # Just drop this event on the floor
            logger.warning(
                "Event {} expired in the queue - dropping...".format(event_dict))
            continue

        # At this point, we have a somewhat legit event, so let's try to process it
        try:
            recording_url = device.recording_url(event_id)
            if not recording_url:
                logger.info("Recording not ready for event {}".format(event_dict))
                # Don't forget to re-enqueue
                event_queue.put(event_dict)
            else:
                handle_video(recording_url, event, device)

        except Exception as e:
            logger.exception(e)

        # Finally, take a rest...
        time.sleep(WORKER_SLEEP_SECS)


def main():
    configure_logger()
    register_stack_dump()

    global slack
    slack = Slacker(SLACK_API_KEY)

    global worker_thread
    worker_thread = Thread(target=worker_loop, name="Worker Thread")
    worker_thread.daemon = True
    worker_thread.start()

    global s3_client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECREY_KEY,
    )

    logger.info("Connecting to Ring API")
    if cache_file.is_file():
        auth = Auth("MyProject/1.0", loads(cache_file.read_text()), token_updated)
    else:
        username = RING_USERNAME
        password = RING_PASSWORD
        auth = Auth("MyProject/1.0", None, token_updated)
        try:
            auth.fetch_token(username, password)
        except MissingTokenError:
            auth.fetch_token(username, password, otp_callback())

    ring = Ring(auth)
    ring.update_data()
    logger.info("Connected to Ring API")

    devices = ring.devices()
    logger.info("Found devices {}".format(devices))

    video_devices = devices["doorbots"] + devices["stickup_cams"]
    logger.info("Watching for events on devices {}".format(video_devices))

    # Init event ids
    for device in video_devices:
        device.latest_id = get_latest_recording(device)["id"]
        logger.info("Found latest event id {} for device {}".format(
            device.latest_id, device))

    # Loop for ever, checking to see if a new video becomes available
    try:
        while True:
            time.sleep(HIST_POLL_TIMEOUT_SECS)

            for device in video_devices:
                event = get_latest_recording(device)
                new_id = event["id"]

                if new_id != device.latest_id:
                    logger.info("Detected a new event for device {}".format(device))
                    device.latest_id = new_id
                    enqueue_event(event, device)

    except KeyboardInterrupt:
        exit(0)


def dump_stack(sig, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    logger.error("Stack Dump:\n{}".format("\n".join(code)))


def register_stack_dump():
    signal.signal(signal.SIGUSR1, dump_stack)  # Register handler


if __name__ == "__main__":
    main()
