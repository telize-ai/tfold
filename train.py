from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import io
import os

import tensorflow as tf
from PIL import Image
from google.protobuf import text_format

from config.config import LabelMapConfig, DetectionConfig
from object_detection import exporter
from object_detection import trainer
from object_detection.builders import dataset_builder, model_builder
from object_detection.exceptions import ModelDirectoryExists
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from object_detection.utils import dataset_util

slim = tf.contrib.slim

DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_CHECKPOINT = os.path.join(DIR, 'data', 'checkpoint', 'model.ckpt')
OUTPUT = os.path.join(DIR, 'builds')

# Can vary fro 10k to 300k, to provide acceptable results.
NUM_STEPS = 50000


class Trainer(object):
    def __init__(self, model_name=None, model_dir=None, model_checkpoint=MODEL_CHECKPOINT):
        self.model_name = model_name

        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(OUTPUT, self.model_name)

        self.label_map_config_path = os.path.join(self.model_dir, 'label_map.config')
        self.object_detection_config_path = os.path.join(self.model_dir, 'object_detection.config')
        self.train_input_path = os.path.join(self.model_dir, 'train.record')
        self.eval_input_path = os.path.join(self.model_dir, 'eval.record')

        self.model_checkpoint = model_checkpoint

        self.classes = {}
        self.train_samples = []
        self.eval_samples = []

    def prepare(self):
        if os.path.exists(self.model_dir):
            raise ModelDirectoryExists

        os.mkdir(self.model_dir)

        label_map_config = LabelMapConfig(items=self.items)
        label_map_config.write(self.label_map_config_path)

        object_detection_config = DetectionConfig(
            num_classes=self.num_classes,
            fine_tune_checkpoint=self.model_checkpoint,
            train_input_path=self.train_input_path,
            eval_input_path=self.eval_input_path,
            label_map_path=self.label_map_config_path,
            num_steps=NUM_STEPS
        )
        object_detection_config.write(self.object_detection_config_path)
        self.write_tf_records(self.train_input_path, self.train_samples)
        self.write_tf_records(self.eval_input_path, self.eval_samples)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def items(self):
        _items = []
        for class_name, class_id in self.classes.items():
            _items.append({
                'name': class_name.lower(),
                'id': class_id,
                'display_name': class_name
            })
        return _items

    def get_or_create_class(self, class_text):
        if class_text not in self.classes:
            classes_len = len(self.classes)
            self.classes[class_text] = classes_len + 1
        return self.classes[class_text]

    def add_train_sample(self, filename, boxes):
        """
        :param filename: "test_images/image1.jpg",
        :param boxes: [
                {
                    "xmin": 10,
                    "xmax": 20,
                    "ymin": 10,
                    "ymax": 20,
                    "class_text": "test"
                }
            ]
        """
        for box in boxes:
            box['class_int'] = self.get_or_create_class(box['class_text'])
        self.train_samples.append({'filename': filename, 'boxes': boxes})

    def add_eval_sample(self, filename, boxes):
        """
        :param filename: "test_images/image1.jpg",
        :param boxes: [
                {
                    "xmin": 10,
                    "xmax": 20,
                    "ymin": 10,
                    "ymax": 20,
                    "class_text": "test"
                }
            ]
        """
        print(boxes)
        for box in boxes:
            box['class_int'] = self.get_or_create_class(box['class_text'])
        self.eval_samples.append({'filename': filename, 'boxes': boxes})

    @staticmethod
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

    @staticmethod
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
            tf_example = Trainer.create_tf_example(sample)
            writer.write(tf_example.SerializeToString())
        writer.close()

    def start(self):
        configs = config_util.get_configs_from_pipeline_file(self.object_detection_config_path)
        model_config = configs['model']
        train_config = configs['train_config']
        input_config = configs['train_input_config']

        model_fn = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=True)

        def get_next(config):
            return dataset_builder.make_initializable_iterator(
                dataset_builder.build(config)).get_next()

        create_input_dict_fn = functools.partial(get_next, input_config)

        ps_tasks = 0
        worker_replicas = 1
        worker_job_name = 'lonely_worker'
        task = 0
        is_chief = True
        master = ''
        num_clones = 1
        clone_on_cpu = False
        graph_rewriter_fn = None

        try:
            trainer.train(
                create_input_dict_fn,
                model_fn,
                train_config,
                master,
                task,
                num_clones,
                worker_replicas,
                clone_on_cpu,
                ps_tasks,
                worker_job_name,
                is_chief,
                self.model_dir,
                graph_hook_fn=graph_rewriter_fn
            )
        except KeyboardInterrupt:
            pass

        self.export_inference()

    def get_checkpoint(self):
        checkpoint_prefix = glob.glob(self.model_dir + '/model.ckpt-*.index')[-1]
        checkpoint_prefix = checkpoint_prefix.replace('.index', '')
        return checkpoint_prefix

    def export_inference(self):

        model_config = os.path.join(self.model_dir, 'object_detection.config')
        checkpoint_prefix = self.get_checkpoint()
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        output_dir = os.path.join(self.model_dir, 'build')
        write_inference_graph = False
        with tf.gfile.GFile(model_config, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)

        input_shape = None
        exporter.export_inference_graph(
            input_type='image_tensor',
            pipeline_config=pipeline_config,
            trained_checkpoint_prefix=checkpoint_prefix,
            output_directory=output_dir,
            input_shape=input_shape,
            write_inference_graph=write_inference_graph
        )


if __name__ == '__main__':
    """
       The flow:
           train = Train(mode_name='brands', model_dir='./my_models/brands')
           for filename, boxes in train_samples:
               train.add_train_sample(filename, boxes)
           for filename, boxes in eval_samples:
               train.add_eval_sample(filename, boxes)

           train.prepare()
           train.start()
       """

    import json

    data_dir = '../telize_app/tmp/export'

    json_data = open(os.path.join(data_dir, 'samples.json')).read()
    samples = json.loads(json_data)

    train = Trainer(model_name='sport_brands', model_dir='tmp/sport_brands')

    for sample in samples['train']:
        filename = os.path.join(data_dir, sample['filename'])
        train.add_train_sample(filename, sample['boxes'])

    for sample in samples['eval']:
        filename = os.path.join(data_dir, sample['filename'])
        train.add_eval_sample(filename, sample['boxes'])

    train.prepare()
    train.start()
