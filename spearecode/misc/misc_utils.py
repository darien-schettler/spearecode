import os
import re
import pickle
import random
import numpy as np
import tensorflow as tf


def prevent_mask_scaling(logits):
    try:
        # Drop the keras mask, so it doesn't scale the losses/metrics --> b/250038731
        del logits._keras_mask
    except AttributeError:
        pass

    return logits