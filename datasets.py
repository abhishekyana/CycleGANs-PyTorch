import numpy as np
import torch.vision.transforms as transforms
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from scipy.misc import imresize


class A2BDataset(Dataset):
	def __init__(self, path, mode='train', transforms_arg=None, unpaired=False, size=(256,256,3)):
		self.transforms = transforms.Compose(transforms_arg)
		self.unpaired = unpaired
		self.size = size
		self.A_paths = sorted(glob.glob(f'{root}/{mode}/A/*.*'))
		self.B_paths = sorted(glob.glob(f'{root}/{mode}/B/*.*'))
		self.totA_paths = len(self.A_paths)
		self.totB_paths = len(self.B_paths)
		self.cache = {'A':{}, 'B':{}}
	
	def __getitem__(self, idx):
		idxa = idx%self.totA_paths
		if idxa in self.cache['A']:
			image_A = self.cache['A'][idxa]
		else:
			imageA = Image.open(self.A_paths[idxa])
			imageA = imresize(imageA, (256,256,3))
			imageA = np.array(imageA)
			if len(imageA.shape)==2: #For mono channel images
				imageA = np.stack((imageA,)*3, axis=-1)
			imageA = Image.fromarray(imageA)
			image_A = self.transforms(imageA)
			self.cache['A'][idxa] = image_A

		if self.unpaired:
			idxb = np.random.randint(0, self.totB_paths-1)
		else:
			idxb = idx%self.totB_paths

		if idxb in self.cache['B']:
			image_B = self.cache['B'][idxb]
		else:
			imageB = Image.open(self.B_paths[idxb])
			imageB = imresize(imageB, (256,256,3))
			imageB = np.array(imageB)
			if len(imageB.shape)==2: #For mono channel images
				imageB = np.stack((imageB,)*3, axis=-1)
			imageB = Image.fromarray(imageB)
			image_B = self.transforms(imageB)
			self.cache['B'][idxb] = image_B
		return {'A':image_A, 'B':image_B}