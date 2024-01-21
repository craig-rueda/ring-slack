import io
import os

from setuptools import find_packages, setup

VERSION = '0.0.1'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with io.open('README.md') as f:
    long_description = f.read()

setup(
    name='ringer',
    description=('Pumps Ring events into a Slack channel'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "apng==0.3.3",
        "boto3==1.34.23",
        "opencv-contrib-python==4.9.0.80",
        "opencv-python-headless==4.9.0.80",
        "Pillow==10.2.0",
        "pyaml==23.12.0",
        "ring_doorbell[listen]==0.8.5",
        "slacker==0.13.0",
    ],
    author='Craig Rueda',
    author_email='craig@craigrueda.com',
    url='https://craigrueda.com',
    download_url=(
        'https://github.com/craig-rueda/ring-slack/' + VERSION
    ),
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
    tests_require=[
        'nose>=1.0',
    ],
    test_suite='nose.collector',
    entry_points = {
        'console_scripts': [
            'ring-face = ringer.cli:main',
        ],
    },
)
