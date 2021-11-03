from keras import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Rescaling, GlobalAveragePooling2D, Dense
from keras.metrics import CategoricalAccuracy
import logging

from keras.optimizer_v2.rmsprop import RMSprop

logger = logging.getLogger()

def build_model(input_shape, weights, n_classes):
    model = MobileNetV2(input_shape=input_shape, weights='imagenet' if not weights else None, include_top=False)
    # freeze feature extraction layers
    model.trainable = False
    # assemble the model from: pre-processing layers, top layers of MobileNetv2, classification layers
    res_model = Sequential(
        [Rescaling(scale=2, offset=-1, input_shape=input_shape),
         model,
         GlobalAveragePooling2D(), Dense(n_classes, activation='softmax', name='predictions')])
    res_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001),
                      metrics=[CategoricalAccuracy()])
    if weights:
        logger.info('Loading weights...')
        res_model.load_weights(weights)
    return res_model

def preprocess_input(x):
    return x/255.