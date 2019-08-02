import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
	def __init__(self, inFeatures):
		super(ResBlock, self).__init__()
		self.conv = nn.Sequential(nn.ReflectionPad2d(1), 
									nn.Conv2d(inFeatures, inFeatures, 3),
									nn.InstanceNorm2d(inFeatures),
									nn.ReLU(inplace=True),
									nn.ReflectionPad2d(1),
									nn.Conv2d(inFeatures, inFeatures, 3),
									nn.InstanceNorm2d(inFeatures))
	def forward(self, X):
		out = X + self.conv(X)
		return out


class Generator(nn.Module):
	def __init__(self, inputnc, outputnc, nResBlocks=9):
		super(Generator, self).__init__()

		layers = [nn.ReflectionPad2d(3),
						  nn.Conv2d(inputnc, 64, 7),
						  nn.InstanceNorm2d(64),
						  nn.ReLU(inplace=True)]
		#To downsample the Image
		inFeatures = 64
		outFeatures = 2*inFeatures

		for i in range(2):
			layers += [nn.Conv2d(inFeatures, outFeatures, 3, stride=2, padding=1),
									 nn.InstanceNorm2d(outFeatures),
									 nn.ReLU(inplace=True)]
			inFeatures = outFeatures
			outFeatures = 2*inFeatures

		for i in range(nResBlocks):
			layers += [ResBlock(inFeatures)]

			#To upsample the Image
		outFeatures = inFeatures//2

		for i in range(2):
			layers += [nn.ConvTranspose2d(inFeatures, outFeatures, 3, stride=2, padding=1, output_padding=1),
											nn.InstanceNorm2d(outFeatures),
											nn.ReLU(inplace=True)]
			inFeatures = outFeatures
			outFeatures = inFeatures//2

		layers += [nn.ReflectionPad2d(3),
							 nn.Conv2d(64, outputnc, 7),
							 nn.Tanh()]
		self.model = nn.Sequential(*layers)

	def forward(self, X):
		out=self.model(X)
		return out

class Discriminator(nn.Module):
	def __init__(self, inputnc):
		super(Discriminator, self).__init__()
		layers = [nn.Conv2d(inputnc, 64, 4, stride=2, padding=1),
									nn.LeakyReLU(0.2, inplace=True),
									nn.Conv2d(64, 128, 4, stride=2, padding=1),
									nn.InstanceNorm2d(128),
									nn.LeakyReLU(0.2, inplace=True),
									nn.Conv2d(128, 256, 4, stride=2, padding=1),
									nn.InstanceNorm2d(256), 
									nn.LeakyReLU(0.2, inplace=True),
									nn.Conv2d(256, 512, 4, padding=1),
									nn.InstanceNorm2d(512), 
									nn.LeakyReLU(0.2, inplace=True),
									nn.Conv2d(512, 1, 4, padding=1)]
		self.model = nn.Sequential(*layers)
	def forward(self, X):
		out = self.model(X)
		out = F.avg_pool2d(out, out.size()[2:]).view(out.size()[0], -1)
		return out