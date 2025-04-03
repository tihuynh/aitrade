import os.path

from pybit.unified_trading import HTTP
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

model = Sequential()
print(dir(model))  # Xem các phương thức có sẵn