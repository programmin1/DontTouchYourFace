#!/usr/bin/env python3
# Basically https://gluon-cv.mxnet.io/build/examples_pose/demo_simple_pose.html

import numpy as np
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get a Simple Pose model trained with input images of size 256x192 on MS COCO
# dataset. We pick the one using ResNet-18 V1b as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# Note that a Simple Pose model takes a top-down strategy to estimate
# human pose in detected bounding boxes from an object detection model.

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])

######################################################################
# Pre-process an image for detector, and make inference
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image to 512 px. But you can
# feed an arbitrarily sized image.
#
# This function returns two results. The first is a NDArray with shape
# ``(batch_size, RGB_channels, height, width)``. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true',
                          path='soccer.png')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxs = detector(x)

######################################################################
# Process tensor from detector to keypoint network
# --------------------
#
# Next we process the output from the detector.
#
# For a Simple Pose network, it expects the input has the size 256x192,
# and the human is centered. We crop the bounding boxed area
# for each human, and resize it to 256x192, then finally normalize it.
#
# In order to make sure the bounding box has included the entire person,
# we usually slightly upscale the box size.

pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

######################################################################
# Predict with a Simple Pose network
# --------------------
#
# Now we can make prediction.
#
# A Simple Pose network predicts the heatmap for each joint (i.e. keypoint).
# After the inference we search for the highest value in the heatmap and map it to the
# coordinates on the original image.

predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

def dist(np1, np2):
	print(np1)
	print(np2)
	#HOW do you get a SIMPLE NUMBER result? alsways [number] array
	print('dist %s' % ((np.sum(np.linalg.norm(np1-np2))).tolist(),))
	return np.linalg.norm(np1-np2).item(0)

for person in (pred_coords):
	#print( person )
	#print(person[0])
	hand1 = person[9][0]
	hand2 = person[10]
	leg1 = person[15]
	leg2 = person[16]
	print('feet:')
	print(leg1)
	print(leg2)
	print('hands:')
	print(hand1)
	print(hand2)
	#rough size of head.
	head1 = person[0]
	headdist = dist(head1,person[1])
	for i in range(4):
		d = dist(head1,person[i])
		if d > headdist:
			headdist = d
	print('headdist about %s' % (headdist,))
	#Is hand anywhere near head points - within one head distance in pixels, near any head point?
	close = False
	for i in range(4):
		if dist(hand1,person[i]) < headdist*2:
			close = True
		if dist(hand2,person[i]) < headdist*2:
			close = True
	if close:
		print("Person touching face")
	else:
		print("Person not touching face")

######################################################################
# Display the pose estimation results
# ---------------------
#
# We can use :py:func:`gluoncv.utils.viz.plot_keypoints` to visualize the
# results.

ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
plt.show()
