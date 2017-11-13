from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datasets import dataset_utils

_NCORE =  20

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

def process(q, iolock):
    while True:
        d = q.get()
        if d is None:
            break
        product_id = d['_id']
        category_id = d['category_id']
        prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            # do something with the picture, etc

def run(dataset_dir):
    """Runs the download and conversion operation.

      Args:
        dataset_dir: The dataset directory where the dataset is stored.
      """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    cat2idx, idx2cat = _make_category_tables(dataset_dir)


    pass