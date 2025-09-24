import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import copy

# XAI imports
try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore
    XAI_AVAILABLE = False
except ImportError:
    print("tf-keras-vis not available. Install with: pip install tf-keras-vis")
    XAI_AVAILABLE = False
