import math
import tensorflow as tf
import os
import pdb
import numpy as np
from datasets import dataset_factory
from nets import nets_factory
import nets.resnet_v2 as resnet_v2
from preprocessing import preprocessing_factory
slim = tf.contrib.slim

def merge_predictions(predictions_fn):
    '''
    Merge predictions/logit scores for products that are the same.
    '''

    out_f = open(predictions_fn + '_merged', 'w')
    f = open(predictions_fn, 'r')
    line = f.readline().strip().split()
    curr_id = line[0]
    curr_scores = np.power(np.array([float(x) for x in line[1:]]), 3)
    num_elems = 1
    line = f.readline().strip().split()

    while line != []:
        id = line[0]
        # raise elements to the third power, and then take the cubic root
        scores = np.power(np.array([float(x) for x in line[1:]]), 3)

        if id == curr_id:
            num_elems += 1
            curr_scores += scores
        else:
            curr_scores = np.cbrt(curr_scores / float(num_elems))
            curr_scores_str = [str(x) for x in curr_scores]
            out_f.write(curr_id + ' ' + ' '.join(curr_scores_str) + '\n')

            curr_scores = scores
            num_elems = 1
            curr_id = id

        line = f.readline().strip().split()


    curr_scores = np.cbrt(curr_scores / float(num_elems))
    curr_scores_str = [str(x) for x in curr_scores]
    out_f.write(curr_id + ' ' + ' '.join(curr_scores_str) + '\n')

    out_f.close()
    f.close()


if __name__ == '__main__':

    checkpoint_dir = '/home/shunan/Code/Data/cdiscount/training'
    dataset_dir = '/home/shunan/Code/Data/cdiscount/tf_records'
    test_fn = '/home/shunan/Code/Data/cdiscount/tf_records/cdiscount_test.tfrecord'
    num_classes = 5270
    image_size = 180
    batch_size = 100
    set_name = 'test'
    data_sizes = {'train': 12195682, 'validation': 175611, 'test': 3095080}
    out_fn = os.path.join(dataset_dir, '{}_predictions.txt'.format(set_name))

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    # loading the dataset
    dataset = dataset_factory.get_dataset('cdiscount', set_name, dataset_dir)

    # dataset provider to load data from the dataset.
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False, common_queue_capacity=2*batch_size,
        common_queue_min=batch_size)
    [image, label, product_id] = provider.get(['image', 'label', 'product_id'])

    # Pre-processing step.
    image_preprocessing_fn = preprocessing_factory.get_preprocessing('simple', is_training=False)
    image = image_preprocessing_fn(image, image_size, image_size)

    images, labels, product_ids = tf.train.batch([image, label, product_id], batch_size=batch_size, num_threads=1,
                                                 capacity=5 * batch_size)

    # Get the model
    # network_fn = nets_factory.get_network_fn('resnet_v2_152', num_classes=num_classes, is_training=False)
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=0.)):
        logits, end_points = resnet_v2.resnet_v2_152(images, num_classes=num_classes, is_training=False)

    #Obtain the trainable variables and a saver
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    output_f = open(out_fn, 'w')

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        num_iters = int(math.ceil(data_sizes[set_name] / float(batch_size)))

        for i in range(num_iters):
            output, ids = sess.run([logits, product_ids])
            for j in range(output.shape[0]):
                vec_str = [str(x) for x in output[j, :]]
                output_f.write(str(ids[j]) + ' ' + ' '.join(vec_str) + '\n')

        output_f.close()