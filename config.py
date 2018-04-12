import argparse
import sys
import tensorflow as tf

# import your network file
from nets import rnn


"""
Default Configuration
"""
NUM_GPUS = int(2)


# load your own model
MODEL_NETWORK     = rnn.RNNSampleModel()
MODEL_SAVE_FOLDER = "/mnt/TrainSave_sam/"
MODEL_SAVE_NAME   = "rnn_base"
MODEL_OPTIMIZER   = tf.train.AdamOptimizer
MODEL_SAVE_INTERVAL = int(100)


BATCH_SIZE      = int(128)
IMAGE_WIDTH     = int(28)
IMAGE_HEIGHT    = int(28)
IMAGE_CHANNELS  = int(1)
NUM_CLASS = int(10)



TRAIN_DATA_SHUFFLE  = True
TRAIN_MAX_STEPS     = int(99999999)
TRAIN_LEARNING_RATE = float(0.001)
TRAIN_DECAY         = float(0.1)
TRAIN_WEIGHT_DECAY = float(0.0001)
TRAIN_DECAY_INTERVAL= float(1050000)





