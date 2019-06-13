import os
import shutil
import tempfile

import cv2
import logging

import requests
from apng import APNG
from ring_doorbell import Ring, RingDoorBell

from .config import RING_PASSWORD, RING_USERNAME

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
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

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
    img_dir = os.path.join(temp_dir, "imgs")
    result_png = os.path.join(temp_dir, "result.png")
    os.mkdir(img_dir)

    try:
        video = cv2.VideoCapture(fetch_video(video_url, temp_dir))
    except Exception as e:
        logger.exception(e)
        return

    try:
        process_motion(video, img_dir)

        # Build an animated Gif
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        image_files.sort()
        if not image_files:
            logger.warning(f"Found no motion in video {video_url}...")
            return

        APNG.from_files(image_files, delay=500).save(result_png)
    except Exception as e:
        logger.exception(e)
    finally:
        shutil.rmtree(temp_dir)
        video.release()

    # Now, post to slack


if __name__ == "__main__":
    ring = Ring(RING_USERNAME, RING_PASSWORD)
    doorbell = RingDoorBell(ring, "Front Door")

    handle_video(doorbell.recording_url(6701814557218175478))
