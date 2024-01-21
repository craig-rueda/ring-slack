import click
import cv2
from ringer.main import (
    configure_logger,
    init_face_recognizer,
    init_ring_client,
    process_faces,
    register_stack_dump,
    train_face_recognizer,
)


@click.group()
def main():
    configure_logger()
    register_stack_dump()


@main.command("train")
def train_classifier():
    train_face_recognizer()


@main.command("login_ring")
def login_ring():
    init_ring_client()


@main.command("process_faces")
@click.option(
    "--video_file",
    required=True,
    help="Location of a video file to test",
)
def process_faces_cmd(video_file: str):
    init_face_recognizer()
    click.echo(f"Faces Detected: {process_faces(cv2.VideoCapture(video_file))}")
