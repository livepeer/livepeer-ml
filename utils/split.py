"""
This script splits the dataset into train and test subsets
"""
import argparse
import itertools
import multiprocessing
import os
import cv2
import shutil
import glob
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def resize_image(args):
    try:
        in_path, out_path, new_size = args
        if not os.path.exists(out_path):
            img = cv2.imread(in_path)
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(out_path, img)
    except:
        logger.exception(f'Error resizing image {in_path}')


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

    if args.out_dir:
        # copy classes file
        links_classes = os.path.join(args.out_dir, 'classes.txt')
        if not os.path.exists(links_classes):
            shutil.copy(classes_dst, links_classes)
        # generate path tuples
        paths = []
        for i, f in enumerate(itertools.chain(train_files, test_files)):
            link_path = os.path.join(args.out_dir, 'train' if i < len(train_files) else 'test', f)
            os.makedirs(os.path.dirname(link_path), exist_ok=True)
            src_file_path = os.path.join(args.data_dir, f)
            paths.append((src_file_path, link_path, args.resize_to))
        if not args.resize_to:
            # just create links
            for src_file_path, link_path, _ in tqdm.tqdm(paths, 'Create links', miniters=len(paths) / 100):
                os.symlink(src_file_path, link_path)
        else:
            # resize images and write files
            pool = multiprocessing.Pool(processes=4)
            list(tqdm.tqdm(pool.imap_unordered(resize_image, paths), total=len(paths), miniters=len(paths) / 100,
                           desc='Resize images'))
        if not args.overwrite:
            # remove all files in destination which weren't among new files
            all_files = list(glob.glob(os.path.join(args.out_dir, 'train/**/*.*'))) + list(
                glob.glob(os.path.join(args.out_dir, 'test/**/*.*')))
            diff_files = set(all_files).difference(set([p[1] for p in paths]))
            if len(diff_files) > 0:
                logger.warning(f'Will delete {len(diff_files)} files in destination which are not present in current split')
                for f in tqdm.tqdm(diff_files, 'Deleting...', miniters=len(diff_files) / 100):
                    os.remove(f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="This script generates train-test lists of files in dataset folder, and corresponding dirs with files or symlinks")
    ap.add_argument('--data_dir', help='Dataset directory', required=True)
    ap.add_argument('--splits_out_dir', required=True, help='List of files output dir')
    ap.add_argument('--out_dir', default='', help='Output dir for links or resized images')
    ap.add_argument('--resize_to', default='',
                    help='Comma-separated target dimensions of images. If specified, script will create image files in out_dir, otherwise it will create symlinks to original files')
    ap.add_argument('--overwrite', default=False, action='store_true')
    ap.add_argument('--test_ratio', default=0.1)
    args = ap.parse_args()
    # convert image size
    if args.resize_to:
        args.resize_to = [int(args.resize_to.split(',')[0]), int(args.resize_to.split(',')[1])]
    # check & prepare dirs
    if not os.path.exists(args.data_dir):
        logger.error(f'Data dir {args.data_dir} does not exist')
        exit(-1)
    if args.out_dir:
        if os.path.exists(args.out_dir) and len(os.listdir(args.out_dir)):
            if not args.overwrite:
                logger.warning(f'Out dir is not empty, will not overwrite existing files')
            else:
                shutil.rmtree(args.out_dir + os.sep)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    if not os.path.exists(args.splits_out_dir):
        os.makedirs(args.splits_out_dir)
    make_splits(args)
