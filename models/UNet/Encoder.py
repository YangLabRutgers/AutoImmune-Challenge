import tensorflow as tf
from tensorflow import keras

class UNet_Encoder(keras.layers):
    def __init__(self):
        super().__init__()
        self.conv