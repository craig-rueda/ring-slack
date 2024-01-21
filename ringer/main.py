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
import numpy as np
import yaml

import boto3
import coloredlogs
import cv2
import requests
from PIL import Image
from apng import APNG
from oauthlib.oauth2 import MissingTokenError
from ring_doorbell import Auth, Requires2FAError, Ring, RingEvent

from .config import (
    AWS_ACCESS_KEY,
    AWS_TRAINING_BUCKET_NAME,
    AWS_TRAINING_BUCKET_PREFIX,
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
    WORKER_SLEEP_SECS,
)

SAVED_TRAINING_DATA_OUT_KEY = "saved_training_output.yaml"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
logger = logging.getLogger(__name__)
event_queue = Queue()
auth_cache_file = Path("test_token.cache")
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_training_data_manifest = {}


def create_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECREY_KEY,
    )


def token_updated(token):
    auth_cache_file.write_text(dumps(token))


def otp_callback():
    return input("2FA code: ")


def init_ring_client():
    logger.info("Connecting to Ring API")
    if auth_cache_file.is_file():
        auth = Auth(USER_AGENT, loads(auth_cache_file.read_text()), token_updated)
    else:
        auth = Auth(USER_AGENT, None, token_updated)
        auth.fetch_token(RING_USERNAME, RING_PASSWORD, otp_callback())

    global ring
    ring = Ring(auth)
    ring.update_data()
    logger.info("Connected to Ring API")


def gobble_frames(video, frames_to_gobble=5):
    check, frame = video.read()
    read_frames = 0
    while check and read_frames < frames_to_gobble:
        check, frame = video.read()
        read_frames += 1

    return read_frames


def load_training_data_manifest():
    logger.info("Loading training manifest")
    s3_client = create_s3_client()
    global face_training_data_manifest
    face_training_data_manifest = yaml.safe_load(
        s3_client.get_object(
            Bucket=AWS_TRAINING_BUCKET_NAME, Key="training_manifest.yaml"
        )["Body"]
    )


def init_face_recognizer():
    load_training_data_manifest()
    s3_client = create_s3_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_file_path = os.path.join(tmp_dir, SAVED_TRAINING_DATA_OUT_KEY)
        s3_client.download_file(
            AWS_TRAINING_BUCKET_NAME, SAVED_TRAINING_DATA_OUT_KEY, local_file_path
        )

        logger.info("Loading saved classifier data...")
        face_recognizer.read(local_file_path)


def train_face_recognizer():
    # Iterate over all training movies and train the recognizer
    s3_client = create_s3_client()
    training_faces_data = []
    training_faces_names = []

    load_training_data_manifest()

    with tempfile.TemporaryDirectory() as tmp_dir:
        idx = -1
        for data in face_training_data_manifest["data"]:
            idx += 1
            for training_file in data["training_files"]:
                file_name = os.path.basename(training_file)
                local_file_path = os.path.join(tmp_dir, file_name)

                # Download each file
                logger.info("Fetching training file %s", training_file)
                s3_client.download_file(
                    AWS_TRAINING_BUCKET_NAME, training_file, local_file_path
                )

                # For each file, train the recognizer
                video_file = os.path.join(tmp_dir, file_name)
                video = cv2.VideoCapture(video_file)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                frame_progress_chunk_len = int(frame_count / 10)
                processed_frames = 0

                while True:
                    result, video_frame = video.read()  # read frames from the video
                    if not result:
                        logger.info(
                            "Training complete for video file %s", training_file
                        )
                        break  # terminate the loop if the frame is not read successfully
                    else:
                        processed_frames += 1

                    if processed_frames % frame_progress_chunk_len == 0:
                        logger.info(
                            "Processed %s%% of training file %s",
                            round((processed_frames / frame_count) * 100),
                            training_file,
                        )

                    # Convert the frame to grayscale
                    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

                    faces = face_classifier.detectMultiScale(
                        gray_image, 1.1, 5, minSize=(40, 40)
                    )
                    if not len(faces):
                        continue

                    # Just take the first face for training
                    x, y, w, h = faces[0]
                    training_faces_data.append(gray_image[y : y + h, x : x + w])
                    training_faces_names.append(idx)

                video.release()

        logger.info(
            "Training face recognizer on %s faces",
            len(face_training_data_manifest["data"]),
        )
        face_recognizer.train(training_faces_data, np.array(training_faces_names))
        saved_training_output = os.path.join(tmp_dir, SAVED_TRAINING_DATA_OUT_KEY)
        face_recognizer.save(saved_training_output)
        logger.info("Uploading saved training data")

        s3_client.upload_file(
            saved_training_output,
            AWS_TRAINING_BUCKET_NAME,
            SAVED_TRAINING_DATA_OUT_KEY,
        )


def process_faces(video) -> set[str]:
    static_back = None
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    motion_count = 0
    found_faces = set()

    while True:
        result, video_frame = video.read()  # read frames from the video
        if not result:
            break  # terminate the loop if the frame is not read successfully

        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

        for x, y, w, h in faces:
            serial, confidence = face_recognizer.predict(gray_image[y : y + h, x : x + w])

            if confidence > 50:
                found_faces.add(face_training_data_manifest["data"][serial]["name"])

    return found_faces


def fetch_video(video_url, temp_dir):
    r = requests.get(video_url)
    video_file = os.path.join(temp_dir, "src.mp4")

    with open(video_file, "wb") as f:
        f.write(r.content)

    return video_file


def handle_video(video_url, event, device):
    logger.info("Processing video for device {}...".format(device))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_file = fetch_video(video_url, tmp_dir)
            video = cv2.VideoCapture(video_file)
            process_faces(video)
    except Exception as e:
        logger.exception(e)
        return
    finally:
        if video:
            video.release()


def configure_logger():
    logging.basicConfig(
        format="%(asctime)s %(name)-5s [%(levelname)s] %(message)s",
        level=logging.INFO,
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
                "Event {} expired in the queue - dropping...".format(event_dict)
            )
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
    global worker_thread
    worker_thread = Thread(target=worker_loop, name="Worker Thread")
    worker_thread.daemon = True
    worker_thread.start()

    init_ring_client()

    logger.info("Initializing facial recognizer")
    init_face_recognizer()

    devices = ring.devices()
    logger.info("Found devices {}".format(devices))

    video_devices = devices["doorbots"] + devices["stickup_cams"]
    logger.info("Watching for events on devices {}".format(video_devices))

    # Init event ids
    for device in video_devices:
        device.last_id = device.last_recording_id
        logger.info("Found latest event id %s for device %s", device.last_id, device)

    # Loop for ever, checking to see if a new video becomes available
    try:
        while True:
            time.sleep(HIST_POLL_TIMEOUT_SECS)

            for device in video_devices:
                event = device.history(1)[0]
                new_id = event["id"]

                if new_id != device.last_id:
                    logger.info("Detected a new event for device {}".format(device))
                    device.last_id = new_id
                    enqueue_event(event, device)

    except KeyboardInterrupt:
        exit(0)


def dump_stack(sig, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    logger.error("Stack Dump:\n{}".format("\n".join(code)))


def register_stack_dump():
    signal.signal(signal.SIGUSR1, dump_stack)  # Register handler
