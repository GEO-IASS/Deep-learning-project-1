from data_utils import load_CIFAR10

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
import tensorflow as tf
import time
import data_helpers

FLAGS = None

# Load the raw CIFAR-10 data
cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

