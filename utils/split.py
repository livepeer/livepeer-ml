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

from utils import logger


def main(args):
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

    if args.links_out_dir:
        # write symlinks
        for i, f in enumerate(itertools.chain(train_files, test_files)):
            link_path = os.path.join(args.links_out_dir, 'train' if i<len(train_files) else 'test', f)
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
    main(args)
