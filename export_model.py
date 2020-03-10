import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
import cv2
import os
import time
import sys
import argparse
import glob

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from planenet_utils import calcPlaneDepths, drawSegmentationImage, drawDepthImage
# from PlaneNet.utils import calcPlaneDepths, drawSegmentationImage, drawDepthImage

from train_planenet import build_graph, parse_args
from tensorflow.python.tools import freeze_graph


WIDTH = 256
HEIGHT = 192

ALL_TITLES = ['PlaneNet']
ALL_METHODS = [('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 0, 2)]

batchSize=1

tf.reset_default_graph()

img_inp = tf.placeholder(tf.float32, shape=[batchSize, HEIGHT, WIDTH, 3], name='image')
training_flag = tf.constant(False, tf.bool)

options = parse_args()
global_pred_dict, _, _ = build_graph(img_inp, img_inp, training_flag, options)

var_to_restore = tf.global_variables()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())


sess = tf.Session(config=config)
sess.run(init_op)
loader = tf.train.Saver(var_to_restore)
path = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = path + '/checkpoint/sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0'
loader.restore(sess, "%s/checkpoint.ckpt"%(checkpoint_dir))


# freeze
#res2 = []
#for op in tf.get_default_graph().get_operations():
#    res2.append(op.__str__())
#print(res2)

print(global_pred_dict.keys())

graph = tf.get_default_graph()  # global_pred_dict['plane']
input_graph_def = graph.as_graph_def()
output_node_names = ['plane_pred']

output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_node_names)

pb_filepath = 'graph_freeze_plane_net_2020.pb'
with tf.gfile.GFile(pb_filepath, 'wb') as f:
    f.write(output_graph_def.SerializeToString())