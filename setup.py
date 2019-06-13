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
    scripts=['ringer/bin/ringer'],
    install_requires=[
        "apng==0.3.3",
        "opencv-contrib-python==3.4.5.20",
        "opencv-python==3.4.5.20",
        "ring-doorbell==0.2.3",
    ],
    author='Craig Rueda',
    author_email='craig@craigrueda.com',
    url='https://craigrueda.com',
    download_url=(
        'https://github.com/craig-rueda/ring-slack/' + VERSION
    ),
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    tests_require=[
        'nose>=1.0',
    ],
    test_suite='nose.collector',
)
