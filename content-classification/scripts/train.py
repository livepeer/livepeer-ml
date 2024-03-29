"""
Model training code
"""
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import logging
import content_classification
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from test import test
from utils.aug_data_generator import AugmentedDataGenerator

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(module)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def train(input_shape, data_dir, logs_dir, batch, epochs, weights, classes, out_dir, val_fraction):
    np.random.seed(1337)

    # create generator, dataset is large enough to skip affine transformations, but color shifts required to prevent
    # the model from learning predominate color in classes
    data_generator = AugmentedDataGenerator(
        per_channel_shift_range=30,
        # swap_channels=True,
        # grayscale=True,
        # shear_range=0.2,
        # zoom_range=0.2,
        # rotation_range=45,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True,
        preprocessing_function=content_classification.preprocess_input,
        validation_split=val_fraction)

    # create train and val data iterators
    train_iter = data_generator.flow_from_directory(os.path.join(data_dir, 'train'),
                                                    target_size=input_shape[:2],
                                                    batch_size=batch,
                                                    class_mode='categorical',
                                                    classes=classes,
                                                    subset='training',
                                                    seed=1337,
                                                    follow_links=True)

    val_iter = data_generator.flow_from_directory(os.path.join(data_dir, 'train'),
                                                  target_size=input_shape[:2],
                                                  batch_size=batch,
                                                  classes=classes,
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  seed=1337,
                                                  follow_links=True)

    # build model with imagenet or pre-defined weights
    logger.info('Creating model...')
    model = content_classification.build_model(input_shape, weights, len(classes))

    logger.info('Training...')
    fit_result = model.fit(x=train_iter,
                           epochs=epochs,
                           validation_data=val_iter,
                           verbose=2)

    df = pd.DataFrame.from_dict(fit_result.history)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    logger.info(f'\n{df}')

    # logger.info('Evaluating...')
    # test(input_shape, classes, os.path.join(data_dir, 'train'), None, model, logs_dir)

    logger.info('Saving...')
    # save keras model for evaluation
    model.save(os.path.join(args.out_dir, 'keras'), include_optimizer=False)

    logger.info('Freezing graph...')
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

    logger.info('Done.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Dataset directory containing: \n'
                                                      '1. train and test dirs, each of which should have a single '
                                                      'subdir per class with samples'
                                                      '2. classes.txt file with one line per class label \n')
    ap.add_argument("--input_shape", default=224, help="Model input shape", type=int)
    ap.add_argument("--batch_size", default=256, help="Batch size", type=int)
    ap.add_argument("--num_epochs", default=2, help="Number of epochs", type=int)
    ap.add_argument("--weights", default='', help="Start weights")
    ap.add_argument("--out_dir", default='../models/current/', help="Train log dir")
    ap.add_argument("--val_fraction", default=0.1, help="How much data to use as validation subset during training")
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
        os.makedirs(args.out_dir)
    # load classes
    classes = pd.read_csv(os.path.join(args.data_dir, 'classes.txt'), header=None)[0].to_list()
    # configure logging
    logs_dir = os.path.join(args.out_dir, 'logs', 'train')
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, 'train.log'), mode='w')
    fh.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d]: %(process)d %(module)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    # tf.compat.v1.disable_eager_execution()
    #  train
    train((args.input_shape, args.input_shape, 3), args.data_dir, logs_dir, args.batch_size, args.num_epochs, args.weights,
          classes,
          args.out_dir,
          args.val_fraction)
