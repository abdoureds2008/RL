import numpy as np
import pandas as pd
import argparse
import os
import pickle
import time
from datetime import datetime
from collections import deque

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.summary import create_file_writer

# For continuous agent (PPO)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Hyperparameter optimization
import optuna

# SHAP for interpretability
import shap

# Wavelet transforms (if installed)
try:
    import pywt
except ImportError:
    pywt = None

# Sklearn for scaling / dimension reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configured for memory growth.")
        except RuntimeError as e:
            print(e)
