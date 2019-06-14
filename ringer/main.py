import logging
import os
import shutil
import tempfile
import time
import uuid

import boto
import cv2
import requests
from PIL import Image
from apng import APNG
from boto.s3.key import Key
from ring_doorbell import Ring, RingDoorBell
from slacker import Slacker

from .config import (
    AWS_ACCESS_KEY,
    AWS_BUCKET_NAME,
    AWS_SECREY_KEY,
    HIST_POLL_TIMEOUT_SECS,
    RING_PASSWORD,
    RING_USERNAME,
    SLACK_API_KEY,
    SLACK_CHANNEL,
    SLACK_USER,
    SLACK_USERNAME
)

MAX_MOTION_CNT = 10
logger = logging.getLogger(__name__)


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
            logger.info(f"End of input read {frames_read} of {frame_count} frames")
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
            img_location = os.path.join(temp_dir, f"{motion_count:03}.png")
            cv2.imwrite(img_location, frame)
            motion_count += 1

        frames_read += gobble_frames(video)


def fetch_video(video_url, temp_dir):
    r = requests.get(video_url)
    video_file = os.path.join(temp_dir, "src.mp4")

    with open(video_file, "wb") as f:
        f.write(r.content)

    return video_file


def handle_video(video_url):
    temp_dir = tempfile.mkdtemp()
    img_full_dir = os.path.join(temp_dir, "full")
    img_thumb_dir = os.path.join(temp_dir, "thumb")
    result_thumb_png = os.path.join(img_thumb_dir, "result.png")
    os.mkdir(img_full_dir)
    os.mkdir(img_thumb_dir)

    try:
        video = cv2.VideoCapture(fetch_video(video_url, temp_dir))
    except Exception as e:
        logger.exception(e)
        return

    try:
        process_motion(video, img_full_dir)

        image_full_files = [os.path.join(img_full_dir, f) for f in
                            os.listdir(img_full_dir)]
        image_full_files.sort()
        if not image_full_files:
            logger.warning(f"Found no motion in video {video_url}...")
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
        thumb_url = upload_to_s3(result_thumb_png, "thumb", key)  # Thumb version

        # Now, post to slack
        post_to_slack("Something happened", "Movement detected", video_url, thumb_url)

    except Exception as e:
        logger.exception(e)
    finally:
        shutil.rmtree(temp_dir)
        video.release()


def get_latest_recording(doorbell):
    history = doorbell.history(limit=1)
    last_id = 0
    if history:
        last_id = history[0]["id"]

    return last_id


def upload_to_s3(file, folder, key):
    conn = boto.connect_s3(AWS_ACCESS_KEY, AWS_SECREY_KEY)
    bucket = conn.get_bucket(AWS_BUCKET_NAME)
    k = Key(bucket)
    k.storage_class = "STANDARD_IA"
    k.key = f"{folder}/{key}"
    k.set_contents_from_filename(file, headers={'Content-Type': 'image/png'})
    return f"https://s3.amazonaws.com/{AWS_BUCKET_NAME}/{k.name}"


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


if __name__ == "__main__":
    ring = Ring(RING_USERNAME, RING_PASSWORD)
    doorbell = RingDoorBell(ring, "Front Door")
    slack = Slacker(SLACK_API_KEY)

    latest_id = get_latest_recording(doorbell)

    # Loop for ever, checking to see if a new video becomes available
    try:
        while True:
            time.sleep(HIST_POLL_TIMEOUT_SECS)
            new_id = get_latest_recording(doorbell)

            if new_id != latest_id:
                latest_id = new_id
                handle_video(doorbell.recording_url(latest_id))
    except KeyboardInterrupt:
        pass
