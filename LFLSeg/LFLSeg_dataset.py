import torch
import torch.utils.data as data
from torchvision import transforms, utils
from PIL import Image
import numpy as np

class LFLSegDataset(data.Dataset):
	"""
	LFLSeg Dataset
	Class: ['full_leaf': 0, 'partial_leaf': 1, 'non_leaf': 2]
	"""
	def __init__(self, txt_path, transform=None):
		"""
		Args:
			txt_path (string): Path to the txt file with annotations.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		with open(txt_path, encoding="utf-8") as f:
			listpath = f.readlines()
		
		listpath = [x.strip() for x in listpath]
		samples = []

		# label_list = []
		for line in listpath:
			filepath, label = line.split(", ")
			filepath = filepath.encode('utf-8')
			samples.append((filepath, label))
			
		self.samples = samples

		self.transform = transform

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		filepath, label = self.samples[idx]
		img = Image.open(filepath)
		# If an image is grayscale, convert to RGB
		if(img.mode != 'RGB'):
			img = img.convert('RGB')
		
		if self.transform:
			img = self.transform(img)
			
		label = torch.tensor(int(label))

		return img, label