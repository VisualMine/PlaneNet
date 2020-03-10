import os
import tempfile

import time

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('frozen_graph_path', None, 'Path to frozen deeplab graph.')
flags.DEFINE_string('test_images_dir', None, 'Directory containing test images.')
flags.DEFINE_string('output_dir', None, 'Directory to output visualisation results.')

VIS_WIDTH = 256
VIS_HEIGHT = 192

# global model
MODEL = None

class PlaneNetModel(object):
  """Class to load planenet model and run inference."""

  INPUT_TENSOR_NAME = 'image:0'
  OUTPUT_TENSOR_NAME = 'plane_pred:0'
  INPUT_WIDTH = 256
  INPUT_HEIGHT = 192

  def __init__(self, frozen_graph_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    
    image_inp = np.expand_dims(cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT)), 0)
    image_inp = image_inp.astype(np.float32) / 255 - 0.5

    batch_plane_predict = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: image_inp})

    plane_predict = batch_plane_predict[0]
    return resized_image, plane_predict

def run_visualization(img_path):
  global MODEL  
  """Inferences Planenet model and visualizes result."""
  original_im = cv2.imread(img_path)

  print('running Planenet on image %s...' % img_path)
  resized_im, plane_pred = MODEL.run(original_im)

  #resized_im = np.array(resized_im)
  #seg_image = label_to_color_image(seg_map).astype(np.uint8)
  
  #resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)
  #vis_img = cv2.addWeighted(resized_im,1.0,seg_image,1,0)


  #basename = os.path.basename(img_path).split('.')[0]
  #vis_path = os.path.join(FLAGS.output_dir, basename+'-vis.jpg')

  #cv2.imwrite(vis_path, vis_img)

def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)

  global MODEL
  MODEL = PlaneNetModel(FLAGS.frozen_graph_path)

  # get images
  all_img_names = tf.gfile.Glob(os.path.join(FLAGS.test_images_dir, '*.jpg'))

  for f in all_img_names:
    starttick = int(round(time.time() * 1000))
    run_visualization(f)
    elapsed = int(round(time.time() * 1000)) - starttick
    print("  duration {}".format(elapsed))

if __name__ == '__main__':
  tf.app.run()