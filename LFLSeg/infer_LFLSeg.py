# This code is modified based on the code of Kazuto Nakashima (http://kazuto1011.github.io)

import os
import argparse
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import sys
sys.path.append("../models/")
from grad_cam import GradCAM

def save_gradcam(save_path, gcam, raw_image, threshold=0.35, is_segment=False):
	raw_image = np.asarray(raw_image)[:, :, ::-1].copy()
	raw_image = cv2.resize(raw_image, (256, 256))
	h, w, _ = raw_image.shape
	gcam = cv2.resize(gcam, (w, h))

	# Segment the leaf with the given threshold
	if(is_segment):
		background_mask = 1.0-(gcam>=threshold)
		foreground_mask =  gcam>=threshold

		### Uncomment to get the background
		# background_mask = np.stack((background_mask, background_mask, background_mask), axis=2)
		# background = background_mask * raw_image
		# background = background / background.max() * 255.0

		foreground_mask = np.stack((foreground_mask, foreground_mask, foreground_mask), axis=2)
		foreground = foreground_mask * raw_image
		foreground = foreground / foreground.max() * 255.0

		cv2.imwrite(save_path, np.uint8(foreground))
	
	else: # Just output the GradCAM heatmap with given threshold
		gcam = gcam*(gcam>=threshold)
		gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
		gcam = gcam.astype(np.float) + raw_image.astype(np.float)
		gcam = gcam / gcam.max() * 255.0
		cv2.imwrite(save_path, np.uint8(gcam))

	print('Saved result at: %s' % save_path)
	


model_names = sorted(
	name for name in models.__dict__
	if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help='Input the image path')
parser.add_argument("--output", type=str, default='output', help='where to save the result')
parser.add_argument("--cuda", default=True, help='Use GPU or CPU?')
parser.add_argument("--threshold", type=float, default=0.35, help='threshold value (between 0.0 to 1.0) for segmentation')
parser.add_argument("--segment", action='store_true', help='Segment the leaf or not')
parser.add_argument("--target_layer", default='layer4.2', help='Target layer of ResNet101 for generating GradCAM')


args = parser.parse_args()

print(args)

def main():

	device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

	if args.cuda:
		current_device = torch.cuda.current_device()
		print('Running on the GPU:', torch.cuda.get_device_name(current_device))
	else:
		print('Running on the CPU')


	# Load the LFLSeg module (ResNet-101 backbone)
	LFLSeg_model = models.resnet101()
	num_ftrs = LFLSeg_model.fc.in_features
	LFLSeg_model.fc = nn.Linear(num_ftrs, 3) # Replace final layer with 3 outputs (full leaf, partial leaf, non-leaf)

	# Dowload the pretrained model: https://drive.google.com/drive/folders/1HqBYjUGXxl1eAkzhURoV5JAqWHvBvvTp?usp=sharing
	load_path = '/path/to/LFLSeg_resnet101.pth'
	LFLSeg_model.load_state_dict(torch.load(load_path), strict=True)

	LFLSeg_model.to(device)
	LFLSeg_model.eval()

	# Load the GradCAM function
	gcam = GradCAM(model=LFLSeg_model)
	
	if(os.path.exists(args.input)==False):
		print("The image path doesn't exist!")
		return
	else:
		# If output folder is not exists, create a new one
		if not os.path.exists(args.output):
			os.makedirs(args.output)

		filename = os.path.basename(args.input)

		raw_image = Image.open(args.input).convert('RGB')

		image = transforms.Compose([
			transforms.Resize(size=(224, 224), interpolation=3),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])(raw_image).unsqueeze(0)

		probs, idx = gcam.forward(image.to(device))

		# Only get the heatmap for the "full leaf" class (i.e., idx=0)
		gcam.backward(idx=0)
		output = gcam.generate(target_layer=args.target_layer)

		save_gradcam('{}/{}_gcam.png'.format(args.output, filename[:-4]),
			output, raw_image, threshold=args.threshold, is_segment=args.segment)


if __name__ == '__main__':
	main()