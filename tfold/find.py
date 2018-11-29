import os

import matplotlib

matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
from PIL import Image

from matplotlib import pyplot as plt
from .object_detection.utils import ops as utils_ops
from .object_detection.utils import label_map_util
from .object_detection.utils import visualization_utils as vis_util


class Detector(object):
    def __init__(self, graph_path, label_path):
        self.graph_path = graph_path
        self.label_path = label_path
        self.image_path = None
        self.image_pil = None
        self.image_np = None
        self.detections = None

    def detect_in_file(self, image_path):
        self.image_path = image_path
        self.image_pil = Image.open(self.image_path)
        return self.detect_in_pil(self.image_pil)

    def detect_in_pil(self, image_pil):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        self.image_np = self.load_image_into_numpy_array(image_pil)

        return self.detect_in_np(self.image_np)

    def detect_in_np(self, image_np):

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        # detections = run_inference_for_single_image(image_np, detection_graph)
        detections = self.run_inference_for_single_image(image_np_expanded)
        return detections

    @property
    def graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    @property
    def categories(self):
        category_index = label_map_util.create_category_index_from_labelmap(
            self.label_path, use_display_name=True)
        return category_index

    @staticmethod
    def load_image_into_numpy_array(img):
        (im_width, im_height) = img.size
        return np.array(img.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, img):
        with self.graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Re-frame is required to translate mask from box coordinates
                    # to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, img.shape[0], img.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(img, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

            self.detections = output_dict
        return output_dict

    def draw_vis(self, output=None, threshold=0.5):
        # Size, in inches, of the output images.

        if not output and self.image_path:
            output = os.path.splitext(self.image_path)[0] + '_detection.png'

        image_size = (12, 8)

        vis_util.visualize_boxes_and_labels_on_image_array(
            self.image_np,
            self.detections['detection_boxes'],
            self.detections['detection_classes'],
            self.detections['detection_scores'],
            self.categories,
            instance_masks=self.detections.get('detection_masks'),
            use_normalized_coordinates=True,
            min_score_thresh=threshold,
            line_thickness=8)
        plt.figure(figsize=image_size)
        plt.imshow(self.detector.image_np)
        plt.savefig(output)
        return output


def main():
    test_images = [os.path.join('test_images', 'motan_{}.jpeg'.format(i)) for i in range(1, 12)]

    detector = Detector(
        graph_path='/home/marius/Projects/tfold/tmp/bak-sport_bransd-6/build/frozen_inference_graph.pb',
        label_path='/home/marius/Projects/tfold/tmp/bak-sport_bransd-6/label_map.config'
    )

    for image_path in test_images:
        detections = detector.detect_in_file(image_path)

        # Visualization of the results of a detection.
        detector.draw_vis(threshold=0.4)


if __name__ == '__main__':
    main()
