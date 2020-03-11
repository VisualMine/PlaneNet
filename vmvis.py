import os
import tempfile

import time

import cv2
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
from PIL import Image

import tensorflow as tf

import planenet_utils

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
  OUTPUT_TENSOR_NAME = ['plane_pred:0', 'segmentation_pred:0', 'non_plane_mask_pred:0', 'non_plane_depth_pred:0', 'non_plane_normal_pred:0']
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
    
    resized_image = cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
    image_inp = np.expand_dims(resized_image, 0)
    image_inp = image_inp.astype(np.float32) / 255 - 0.5

    global_pred = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: image_inp})

    pred_p = global_pred[0][0]
    pred_s = global_pred[1][0]    
    pred_np_m = global_pred[2][0]
    pred_np_d = global_pred[3][0]
    pred_np_n = global_pred[4][0]

    return resized_image, pred_p, pred_s, pred_np_m, pred_np_d, pred_np_n


def run_visualization(img_path):
  global MODEL  
  """Inferences Planenet model and visualizes result."""
  img_ori = cv2.imread(img_path)

  print('running Planenet on image %s...' % img_path)
  resized_im, pred_p, pred_s, pred_np_m, pred_np_d, _ = MODEL.run(img_ori)

  all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)

  info = np.zeros(20)

  focalLength = planenet_utils.estimateFocalLength(img_ori)
  info[0] = focalLength
  info[5] = focalLength
  info[2] = img_ori.shape[1] / 2
  info[6] = img_ori.shape[0] / 2
  info[16] = img_ori.shape[1]
  info[17] = img_ori.shape[0]
  info[10] = 1
  info[15] = 1
  info[18] = 1000
  info[19] = 5

  width_high_res = img_ori.shape[1]
  height_high_res = img_ori.shape[0]
  numOutputPlanes = 10

  # output debug segmentation image
  segmentationImage = planenet_utils.drawSegmentationImage(pred_s, blackIndex=numOutputPlanes)
  segmentationImageBlended = (segmentationImage * 0.7 + resized_im * 0.3).astype(np.uint8)
  cv2.imwrite(os.path.join(FLAGS.output_dir, 'blended.png'), segmentationImageBlended)

  # calculate perpixel depth
  plane_depths = planenet_utils.calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

  pred_np_d = np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)
  all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

  all_segmentations = np.stack([cv2.resize(all_segmentations[:, :, planeIndex], (width_high_res, height_high_res)) for planeIndex in range(all_segmentations.shape[-1])], axis=2)
      
  segmentation = np.argmax(all_segmentations, 2)
  pred_d = all_depths.reshape(-1, numOutputPlanes + 1)[np.arange(height_high_res * width_high_res), segmentation.reshape(-1)].reshape(height_high_res, width_high_res)

  # output debug depth image
  cv2.imwrite(os.path.join(FLAGS.output_dir, 'depth.png'), planenet_utils.drawDepthImage(pred_d))

  # calculate plane center and world normal
  planeIndex = 5
  mask = (segmentation == planeIndex).astype(np.uint8)
  mask[mask != 0] = 255
  cv2.imwrite(os.path.join(FLAGS.output_dir, 'mask_' + str(planeIndex) + '.png'), mask)
  contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  plane = pred_p[planeIndex]
  planeD = np.linalg.norm(plane)
  planeNormal = plane / planeD
  for contour in contours:
    contour = contour.astype(np.float32)
    u = (contour[:, 0, 0].astype(np.float32) / width_high_res * info[16] - info[2]) / info[0] # cx / fx
    v = -(contour[:, 0, 1].astype(np.float32) / height_high_res * info[17] - info[6]) / info[5] # cy / fy
    ranges = np.stack([u, np.ones(u.shape), v], axis=1)
    depth = planeD / np.dot(ranges, planeNormal)
    XYZ = ranges * np.expand_dims(depth, -1)

    for vertexIndex, uv in enumerate(contour):
      X, Y, Z = XYZ[vertexIndex]
      u, v = uv[0]

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