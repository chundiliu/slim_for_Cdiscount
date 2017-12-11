from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datasets import dataset_utils
import sys
import pdb
import glob
import multiprocessing as mp      # will come in handy due to the size of the data
import bson
from skimage.data import imread   # or, whatever image library you prefer
import io

_IMAGE_HEIGHT = 180
_IMAGE_WIDTH = 180
_TFRECORD_DIC = 'tf_records_all'

def _make_category_tables(dataset_dir):
    categories_path = os.path.join(dataset_dir, "category_names.csv")
    categories_df = pd.read_csv(categories_path, index_col="category_id")

    # Maps the category_id to an integer index. This is what we'll use to
    # one-hot encode the labels.
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

    categories_df.to_csv(os.path.join(dataset_dir, "category_names.csv"))
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def _convert_dataset_train(dataset_dir, output_filename, split_name, cat2idx, _image_count):
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        print("Processing {}".format(split_name))
        data = bson.decode_file_iter(open(split_name, 'rb'))
        for c, d in enumerate(data):
            product_id = d['_id']
            category_id = d['category_id']
            class_id = cat2idx[category_id]
            for e, pic in enumerate(d['imgs']):
                example = dataset_utils.image_to_tfexample(pic['picture'], b'png', _IMAGE_HEIGHT, _IMAGE_WIDTH,
                                                           class_id, product_id)
                tfrecord_writer.write(example.SerializeToString())
                _image_count[0] += 1

def _convert_dataset_eval(dataset_dir, output_filename, split_name, cat2idx, _image_count, is_testing = False):
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        print("Processing {}".format(split_name))
        data = bson.decode_file_iter(open(split_name, 'rb'))
        for c, d in enumerate(data):
            product_id = d['_id']
            if is_testing:
                class_id = -1
            else:
                category_id = d['category_id']
                class_id = cat2idx[category_id]
            for e, pic in enumerate(d['imgs']):
                example = dataset_utils.image_to_tfexample(pic['picture'], b'png', _IMAGE_HEIGHT, _IMAGE_WIDTH,
                                                           class_id, product_id)
                tfrecord_writer.write(example.SerializeToString())
                _image_count[0] += 1

def run(dataset_dir):
    """Runs the download and conversion operation.

      Args:
        dataset_dir: The dataset directory where the dataset is stored.
      """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    cat2idx, idx2cat = _make_category_tables(dataset_dir)

    sys.stdout.write("Converting to tfrecords\n")
    sys.stdout.flush()
    # Convert the training data
    _image_count = [0] #Count how many images in this split, use list to do reference pass
    train_splits_fn = sorted(glob.glob(os.path.join(dataset_dir, "split_train/*.bson")))
    for train_split in train_splits_fn:
        shad = train_split.split('/')[-1].split('.')[0]
        output_filename = os.path.join(dataset_dir,"{}/cdiscount_train_{}.tfrecord".format(_TFRECORD_DIC, shad))
        _convert_dataset_train(dataset_dir, output_filename, train_split, cat2idx, _image_count)
    print("{} images in train split are converted into tfrecords file!".format(_image_count[0]))
    #Convert Validation Data
    _image_count = [0]  # Count how many images in this split, use list to do reference pass
    valid_split = os.path.join(dataset_dir, "valid_split.bson")
    output_filename = os.path.join(dataset_dir,"{}/cdiscount_valid.tfrecord".format(_TFRECORD_DIC))
    _convert_dataset_eval(dataset_dir, output_filename, valid_split, cat2idx, _image_count)
    print("{} images in valid split are converted into tfrecords file!".format(_image_count[0]))
    #Convert Testing Data
    _image_count = [0]  # Count how many images in this split, use list to do reference pass
    test_split = os.path.join(dataset_dir, "test.bson")
    output_filename = os.path.join(dataset_dir, "{}/cdiscount_test.tfrecord".format(_TFRECORD_DIC))
    _convert_dataset_eval(dataset_dir, output_filename, test_split, cat2idx, _image_count, is_testing = True)
    print("{} images in test split are converted into tfrecords file!".format(_image_count[0]))

    print('\nFinished converting the Cdiscount dataset!')
