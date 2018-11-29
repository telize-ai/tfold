# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Tool to export an object detection model for inference.

Prepares an object detection tensorflow graph for inference using model
configuration and a trained checkpoint. Outputs inference
graph, associated checkpoint files, a frozen inference graph and a
SavedModel (https://tensorflow.github.io/serving/serving_basic.html).

The inference graph contains one of three input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [None, None, None, 3]
  * `encoded_image_string_tensor`: Accepts a 1-D string tensor of shape [None]
    containing encoded PNG or JPEG images. Image resolutions are expected to be
    the same if more than 1 image is provided.
  * `tf_example`: Accepts a 1-D string tensor of shape [None] containing
    serialized TFExample protos. Image resolutions are expected to be the same
    if more than 1 image is provided.

and the following output nodes returned by the model.postprocess(..):
  * `num_detections`: Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`: Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores`: Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.
  * `detection_masks`: Outputs float32 tensors of the form
      [batch, num_boxes, mask_height, mask_width] containing predicted instance
      masks for each box if its present in the dictionary of postprocessed
      tensors returned by the model.

Notes:
 * This tool uses `use_moving_averages` from eval_config to decide which
   weights to freeze.

Example Usage:
--------------
python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - inference_graph.pbtxt
 - model.ckpt.data-00000-of-00001
 - model.ckpt.info
 - model.ckpt.meta
 - frozen_inference_graph.pb
 + saved_model (a directory)

Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the second stage post-processing score
threshold to be 0.5):

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"
"""
import tensorflow as tf
from google.protobuf import text_format
import os
from tfold.object_detection import exporter
from tfold.object_detection.protos import pipeline_pb2

slim = tf.contrib.slim


def main(_):

    input_type = 'image_tensor'
    model_dir = 'tmp/sport_brands'
    model_config = os.path.join(model_dir, 'object_detection.config')
    checkpoint_prefix = os.path.join(model_dir, 'model.ckpt-200000')
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    output_dir = os.path.join(model_dir, 'build')
    write_inference_graph = False
    with tf.gfile.GFile(model_config, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    input_shape = None
    exporter.export_inference_graph(
        input_type, pipeline_config, checkpoint_prefix,
        output_dir, input_shape=input_shape,
        write_inference_graph=write_inference_graph)


if __name__ == '__main__':
    tf.app.run()