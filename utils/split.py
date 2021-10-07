"""
This script generates train-test lists of files in dataset folder
"""
import argparse
import itertools
import os
import shutil
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def make_splits(args):
    # create list of sample files
    all_files = glob.glob(args.data_dir + os.sep + '**' + os.sep + '*.*')
    # remove dataset root
    all_files = list(f[len(args.data_dir):] for f in all_files)
    # split
    train_files, test_files = train_test_split(all_files, test_size=args.test_ratio, shuffle=True, random_state=1337)

    # write splits to files
    def write_list(filename, data):
        pd.DataFrame(data).to_csv(filename, header=None, index=False)

    write_list(os.path.join(args.splits_out_dir, 'train.txt'), train_files)
    write_list(os.path.join(args.splits_out_dir, 'test.txt'), test_files)

    # copy or create classes.txt file
    classes_src = os.path.join(args.data_dir, 'classes.txt')
    classes_dst = os.path.join(args.splits_out_dir, 'classes.txt')
    if os.path.exists(classes_src):
        shutil.copy(classes_src, classes_dst)
    else:
        write_list(classes_dst, sorted(pd.unique([f.split(os.sep)[-2] for f in all_files]),
                                       key=lambda x: 'z' if x == 'other' else x))

    if args.links_out_dir:
        # copy classes file
        links_classes = os.path.join(args.links_out_dir, 'classes.txt')
        if not os.path.exists(links_classes):
            shutil.copy(classes_dst, links_classes)
        # write symlinks
        for i, f in enumerate(itertools.chain(train_files, test_files)):
            link_path = os.path.join(args.links_out_dir, 'train' if i < len(train_files) else 'test', f)
            os.makedirs(os.path.dirname(link_path), exist_ok=True)
            src_file_path = os.path.join(args.data_dir, f)
            os.symlink(src_file_path, link_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="This script generates train-test lists of files in dataset folder, and, optionally, corresponding dirs with symlinks to files")
    ap.add_argument('--data_dir', help='Dataset directory', required=True)
    ap.add_argument('--splits_out_dir', required=True)
    ap.add_argument('--links_out_dir', default='')
    ap.add_argument('--overwrite', default=False, action='store_true')
    ap.add_argument('--test_ratio', default=0.1)
    args = ap.parse_args()
    # check & prepare dirs
    if not os.path.exists(args.data_dir):
        logger.error(f'Data dir {args.data_dir} does not exist')
        exit(-1)
    if args.links_out_dir:
        if os.path.exists(args.links_out_dir) and len(os.listdir(args.links_out_dir)):
            if not args.overwrite:
                logger.error(f'Links out dir is not empty, use --overwrite to overwrite it')
                exit(-1)
            else:
                shutil.rmtree(args.links_out_dir + os.sep)
        if not os.path.exists(args.links_out_dir):
            os.makedirs(args.links_out_dir)
    if not os.path.exists(args.splits_out_dir):
        os.makedirs(args.splits_out_dir)
    make_splits(args)
