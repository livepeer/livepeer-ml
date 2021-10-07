"""
Model training code
"""
import argparse
import os
import pandas as pd

import tensorflow as tf
from keras import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, Dense, Reshape
from keras.optimizer_v2.rmsprop import RMSprop
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

import logging

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from utils import freeze_session

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(module)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def build_model(input_shape, weights, n_classes):
    model = MobileNetV2(input_shape=input_shape, weights='imagenet' if not weights else None, include_top=False)
    # freeze feature extraction layers
    model.trainable = False
    # assemble the model
    res_model = Sequential(
        [model, GlobalAveragePooling2D(), Dense(n_classes, activation='softmax', name='predictions')])
    res_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])
    if weights:
        logger.info('Loading weights...')
        res_model.load_weights(weights)
    return res_model


def train(input_shape, data_dir, batch, epochs, weights, classes, out_dir, val_fraction):
    logs_dir = os.path.join(out_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # create generator
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=val_fraction)

    # create train and val data iterators
    train_iter = data_generator.flow_from_directory(os.path.join(data_dir, 'train'),
                                                    target_size=input_shape[::2],
                                                    batch_size=batch,
                                                    class_mode='categorical',
                                                    subset='training')

    val_iter = data_generator.flow_from_directory(os.path.join(data_dir, 'train'),
                                                  target_size=input_shape[::2],
                                                  batch_size=batch,
                                                  class_mode='categorical',
                                                  subset='validation')

    # build model with imagenet or pre-defined weights
    logger.info('Creating model...')
    model = build_model(input_shape, weights, len(classes))

    # training process callbacks
    logging = TensorBoard(log_dir=logs_dir)
    checkpoint = ModelCheckpoint(os.path.join(logs_dir, 'model.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)

    earlystop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='auto')

    logger.info('Training...')
    fit_result = model.fit(x=train_iter, epochs=epochs, validation_data=val_iter,
                           callbacks=[logging, checkpoint, earlystop])

    logger.info('Saving...')

    # save the model as frozen graph for compatibility, input name will be input_1, and output name is Identity

    # convert Keras model to ConcreteFunction
    full_model = tf.function(lambda input_1: model(input_1))
    full_model = full_model.get_concrete_function(input_1=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # save frozen graph from frozen ConcreteFunction to hard drive
    # serialize the frozen graph and its text representation to disk.
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=out_dir,
                      name="tasmodel.pb",
                      as_text=False)

    df = pd.DataFrame.from_dict(fit_result.history)
    df.to_csv(os.path.join(out_dir, 'train_stats.csv'), encoding='utf-8', index=False)
    logger.info('Done.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Dataset directory containing: \n'
                                                      '1. train and test dirs, each of which should have a single '
                                                      'subdir per class with samples'
                                                      '2. classes.txt file with one line per class label \n')
    ap.add_argument("--input_shape", default=224, help="Model input shape", type=int)
    ap.add_argument("--batch_size", default=32, help="Batch size", type=int)
    ap.add_argument("--num_epochs", default=300, help="Number of epochs", type=int)
    ap.add_argument("--weights", default='', help="Start weights")
    ap.add_argument("--out_dir", default='../models/current/', help="Train log dir")
    args = ap.parse_args()
    # check args
    classes_file = os.path.exists(os.path.join(args.data_dir, 'classes.txt'))
    train_dir = os.path.exists(os.path.join(args.data_dir, 'train'))
    test_dir = os.path.exists(os.path.join(args.data_dir, 'test'))
    if not os.path.exists(train_dir) or not os.path.exists(test_dir) \
            or not os.path.exists(classes_file):
        logger.error('data_dir does not exist or has incorrect structure. See --help.')
        exit(-1)
    if args.weights and not os.path.exists(args.weights):
        logger.error('Weights file does not exist')
        exit(-1)
    if not os.path.exists(args.out_dir):
        logger.error('Output dir does not exist')
        exit(-1)
    # load classes
    classes = pd.read_csv(os.path.join(args.data_dir, 'classes.txt'), header=None)[0].to_list()
    # add log file
    logger.addHandler(logging.FileHandler(os.path.join(args.out_dir, 'train.log')))
    # train
    train((args.input_shape, args.input_shape, 3), args.data_dir, args.batch_size, args.num_epochs, args.weights,
          classes,
          args.out_dir)
