from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util


def create_tf_example(sample):
    with tf.gfile.GFile(sample['filename'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = sample['filename'].encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for box in sample['boxes']:
        xmins.append(box['xmin'] / width)
        xmaxs.append(box['xmax'] / width)
        ymins.append(box['ymin'] / height)
        ymaxs.append(box['ymax'] / height)
        classes_text.append(box['class_text'].encode('utf8'))
        classes.append(box['class_int'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def write_tf_records(output_file, samples):
    """
    samples = [
        {
            filename: str,
            boxes: [
                {
                    xmin: int,
                    xmax: int,
                    ymin: int,
                    ymax: int,
                    class_text: str,
                    class_int: int
                },
                ...
            ]
        },
        ...
    ]
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    for sample in samples:
        tf_example = create_tf_example(sample)
        writer.write(tf_example.SerializeToString())
    writer.close()


class Trainer(object):
    def __init__(self, work_dir=None):
        self.work_dir = work_dir

    def add_train_sample(self):
        pass

    def add_test_sample(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    obs_samples = [
        {
            "filename": "test_images/image1.jpg",
            "boxes": [
                {
                    "xmin": 10,
                    "xmax": 20,
                    "ymin": 10,
                    "ymax": 20,
                    "class_text": "test",
                    "class_int": 1
                }
            ]
        }
    ]
    write_tf_records('./test.tf', obs_samples)
