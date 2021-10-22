"""
Model evaluation code
"""
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import logging
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import mobilenet_v2

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(module)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def as_labels(x, classes):
    return list(classes[idx] for idx in np.argmax(x, axis=1))


def test(input_shape, data_dir, model_dir, out_dir):
    np.random.seed(1337)
    batch_size = 32

    # create generator
    data_generator = ImageDataGenerator(
        # dataset is large enough to skip augmentations
        preprocessing_function=None)

    test_iter = data_generator.flow_from_directory(os.path.join(data_dir, 'test'),
                                                   target_size=input_shape[:2],
                                                   class_mode='categorical',
                                                   classes=classes,
                                                   seed=1337,
                                                   follow_links=True,
                                                   batch_size=batch_size,
                                                   shuffle=False)

    # load model
    model = tf.keras.models.load_model(model_dir)

    # load y labels
    y_gt = np.zeros(shape=[len(test_iter.filenames), len(classes)])
    for i, f in enumerate(test_iter.filenames):
        y_gt[i, classes.index(f.split(os.sep)[-2])] = 1

    # evaluate
    results = model.evaluate(test_iter, batch_size=32)
    test_accuracy = results[1]
    test_loss = results[0]
    logger.info(f'Test results:\nloss={test_loss}\naccuracy={test_accuracy}')

    y_pred = model.predict(test_iter)

    y_pred_idx = np.argmax(y_pred, axis=1)
    y_gt_idx = np.argmax(y_gt, axis=1)
    y_pred_labs = as_labels(y_pred, classes)
    y_gt_labs = as_labels(y_gt, classes)

    logger.info(f'Classification report:\n{classification_report(y_gt_labs, y_pred_labs)}')

    cm = pd.DataFrame(
        confusion_matrix(y_pred_labs, y_gt_labs),
        index=classes,
        columns=classes
    )
    logger.info(f'Confusion matrix:\ntrue\pred\n{cm}')

    # write errors
    idx_err = y_pred_idx != y_gt_idx
    y_gt_err = np.array(y_gt_labs)[idx_err]
    y_pred_err = np.array(y_pred_labs)[idx_err]
    files_err = np.array(test_iter.filenames)[idx_err]
    err_df = pd.DataFrame(zip(y_gt_err, y_pred_err), index=files_err, columns=['true_label', 'predicted_label'])
    err_df.to_csv(os.path.join(out_dir, 'errors.csv'))
    return test_loss, test_accuracy


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Dataset directory containing: \n'
                                                      '1. train and test dirs, each of which should have a single '
                                                      'subdir per class with samples'
                                                      '2. classes.txt file with one line per class label \n')
    ap.add_argument("--input_shape", default=224, help="Model input shape", type=int)
    ap.add_argument("--model_dir", default='../models/current/keras/', help="Keras model dir")
    ap.add_argument("--out_dir", default='../models/current/', help="Test output dir")
    ap.add_argument("--use_train", default=False, action='store_true', help="Evaluate metrics on train")
    ap.add_argument("--min_acc", default=0.85, help="Minimum accuracy on test set, affects return code")
    args = ap.parse_args()
    # check args
    classes_file = os.path.exists(os.path.join(args.data_dir, 'classes.txt'))
    test_dir = os.path.exists(os.path.join(args.data_dir, 'test'))
    if not os.path.exists(test_dir) or not os.path.exists(classes_file):
        logger.error('data_dir does not exist or has incorrect structure. See --help.')
        exit(-1)
    if not os.path.exists(args.out_dir):
        logger.error('Output dir does not exist')
        exit(-1)
    # load classes
    classes = pd.read_csv(os.path.join(args.data_dir, 'classes.txt'), header=None)[0].to_list()
    # configure file logging
    fh = logging.FileHandler(os.path.join(args.out_dir, 'test.log'), mode='w')
    fh.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d]: %(process)d %(module)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.info('Start test')
    # run
    test_loss, test_accuracy = test((args.input_shape, args.input_shape, 3), args.data_dir, args.model_dir,
                                    args.out_dir)
    exit(int(test_accuracy < 0.85))
