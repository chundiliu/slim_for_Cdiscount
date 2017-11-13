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


_NCORE =  1
_IMAGE_HEIGHT = 180
_IMAGE_WIDTH = 180

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

def process(q, iolock, tfrecord_writer, cat2idx, sess, image, encoded_png):
    while True:
        d = q.get()
        if d is None:
            break
        # = d['_id']
        category_id = d['category_id']
        class_id = cat2idx[category_id]
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            png_string = sess.run(encoded_png, feed_dict={image: picture})
            height = picture.shape[0]
            width = picture.shape[1]
            pdb.set_trace()
            example = dataset_utils.image_to_tfexample(png_string, b'none', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
            # do something with the picture, etc

def run(dataset_dir):
    """Runs the download and conversion operation.

      Args:
        dataset_dir: The dataset directory where the dataset is stored.
      """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    cat2idx, idx2cat = _make_category_tables(dataset_dir)

    sys.stdout.write("Converting to tfrecords")
    sys.stdout.flush()
    train_splits_fn = sorted(glob.glob(os.path.join(dataset_dir, "split_train/*.bson")))
    i = 1
    for train_slit in train_splits_fn:
        shad = train_slit.split('/')[-1].split('.')[0]
        output_filename = os.path.join(dataset_dir,"tf_records/train_{}.tfrecord".format(shad))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            print("Processing {} of {} train splits".format(i, len(train_splits_fn)))
            data = bson.decode_file_iter(open(train_slit, 'rb'))
            for c, d in enumerate(data):
                category_id = d['category_id']
                class_id = cat2idx[category_id]
                for e, pic in enumerate(d['imgs']):
                    pic['picture']
                    example = dataset_utils.image_to_tfexample(pic['picture'], b'png', _IMAGE_HEIGHT, _IMAGE_WIDTH, class_id)
                    tfrecord_writer.write(example.SerializeToString())
        i += 1
