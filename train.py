import torch
import argparse
import itertools
from PIL import Image
from datasets import AnBDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal

from configure import options #Edit this file to control the parameters once, without everytime pasting huge run command for every run.


inH, inW, inC = options["inResolution"]
outH, outW, outC = options["outResolution"]
print("HEYYY")

if torch.cuda.is_available():
	options["cuda"]=True

def LoadData(path):
	ApplyTransforms = [ transforms.Resize((int(inH*1.12), int(inW*1.12)), Image.BICUBIC),
						transforms.RandomCrop((inH, inW)),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((1/2,)*3,(1/2,)*3)]
	Data = AnBDataset(path, transforms_arg=ApplyTransforms, unpaired=options["unpaired"])
	dataloader = DataLoader(Data, batch_size=options["batchsize"], shuffle=True, num_workers=options["nThreads"])
	return dataloader



def trainer(options):
	startEpoch = options["epoch"]
	nEpochs = options["nEpochs"]
	decayEpoch = options["decayEpoch"]
	assert(nEpochs>decayEpoch), "The decay epoch is larger than total epochs, There will be no decay :P, Sure?"
	#GAN to map from Generate A to B and Discriminate B
	GEN_AtoB = Generator(inC, outC)
	DIS_B = Discriminator(outC)
	
	#GAN to map from Generate B to A and Discriminate A
	GEN_BtoA =  Generator(outC, inC)
	DIS_A = Discriminator(inC)
	if options["cuda"]:
		GEN_AtoB.cuda()
		DIS_B.cuda()
		GEN_BtoA.cuda()
		DIS_A.cuda()
	GEN_AtoB.apply(weights_init_normal)
	DIS_B.apply(weights_init_normal)
	GEN_BtoA.apply(weights_init_normal)
	DIS_A.apply(weights_init_normal)
	if options["continued"]:
		print("continuing from a broken position")
		GEN_AtoB.load_state_dict(torch.load("output/GEN_AtoB.pth"))
		DIS_B.load_state_dict(torch.load("output/DIS_B.pth"))
		GEN_BtoA.load_state_dict(torch.load("output/GEN_BtoA.pth"))
		DIS_A.load_state_dict(torch.load("output/DIS_A.pth"))
	# LOSSES
	GANLoss = torch.nn.MSELoss() # ImageA to ImageB distance
	CycleLoss = torch.nn.L1Loss() # Absolute loss
	IdentityLoss  = torch.nn.L1Loss() # ImageA->ImageB->ImageA' distance

	optim_GAN = torch.optim.Adam(list(GEN_AtoB.parameters())+list(GEN_BtoA.parameters()), lr=options["learningrate"], betas=[0.5, 0.999])
	optim_DIS_A = torch.optim.Adam(DIS_A.parameters(), lr=options["learningrate"], betas=[0.5, 0.999])
	optim_DIS_B = torch.optim.Adam(DIS_B.parameters(), lr=options["learningrate"], betas=[0.5, 0.999])

	# LR Scheduler should be here
	lr_scheduler = lambda ep: 1.0 - max(0, ep+startEpoch)/(nEpochs - decayEpoch)

	lr_scheduler_GAN = torch.optim.lr_scheduler.LambdaLR(optim_GAN, lr_lambda=lr_scheduler)
	lr_scheduler_Dis_A = torch.optim.lr_scheduler.LambdaLR(optim_DIS_A, lr_lambda=lr_scheduler)
	lr_scheduler_Dis_B = torch.optim.lr_scheduler.LambdaLR(optim_DIS_B, lr_lambda=lr_scheduler)

	# Tensors Memory Allocation
	batchsize = options["batchsize"]
	if options["cuda"]:
		print("CUDA-fied")
		Tensor = torch.cuda.FloatTensor
	else:
		Tensor = torch.Tensor
	imageA = Tensor(batchsize, inC, inH, inW)
	imageB = Tensor(batchsize, outC, outH, outW)
	targetReal = Variable(Tensor(batchsize).fill_(1.0), requires_grad=False)
	targetFake = Variable(Tensor(batchsize).fill_(0.0), requires_grad=False)

	FakeAHolder = ReplayBuffer() # Check this Check this
	FakeBHolder = ReplayBuffer()

	dataloader = LoadData(options["datapath"])
	logger = Logger(nEpochs, len(dataloader))
	#Actual Training
	for epoch in range(startEpoch, nEpochs):
		for batch_id, batch_data in enumerate(dataloader):
			realA = Variable(imageA.copy_(batch_data['A']))
			realB = Variable(imageB.copy_(batch_data['B']))

			optim_GAN.zero_grad()

			synthB = GEN_AtoB(realB)
			lossIden_BtoB = IdentityLoss(synthB, realB)*5.0

			synthA = GEN_BtoA(realA)
			lossIden_AtoA = IdentityLoss(synthA, realA)*5.0

			fakeB = GEN_AtoB(realA)
			classB = DIS_B(fakeB)
			lossGEN_A2B = GANLoss(classB, targetReal)

			fakeA = GEN_BtoA(realB)
			classA = DIS_A(fakeA)
			lossGEN_B2A = GANLoss(classA, targetReal)

			reconA = GEN_BtoA(fakeB)
			lossCycle_ABA = CycleLoss(reconA, realA)*10.0

			reconB = GEN_BtoA(fakeA)
			lossCycle_BAB = CycleLoss(reconB, realB)*10.0

			lossTotal = lossCycle_BAB + lossCycle_ABA + lossGEN_B2A + lossGEN_A2B + lossIden_AtoA + lossIden_BtoB
			lossTotal.backward()

			optim_GAN.step()
			# Discriminator A
			optim_DIS_A.zero_grad()

			classAreal = DIS_A(realA)
			lossDreal = GANLoss(classAreal, targetReal)

			fakeA = FakeAHolder.push_and_pop(fakeA)
			classAfake = DIS_A(fakeA.detach())
			lossDfake = GANLoss(classAfake, targetFake)

			lossDA = (lossDreal + lossDfake)/2

			lossDA.backward()

			optim_DIS_A.step()

			# Discriminator B
			optim_DIS_B.zero_grad()

			classBreal = DIS_B(realB)
			lossDreal = GANLoss(classBreal, targetReal)

			fakeB = FakeBHolder.push_and_pop(fakeB)
			classBfake = DIS_B(fakeB.detach())
			lossDfake = GANLoss(classBfake, targetFake)
			lossDB = (lossDreal + lossDfake)/2

			lossDB.backward()

			optim_DIS_B.step()
			# Progress report (http://localhost:8097)
			logger.log({'lossTotal': lossTotal, 'lossIdentity': (lossIden_AtoA + lossIden_BtoB), 'lossGAN': (lossGEN_A2B + lossGEN_B2A),
					'lossCycle': (lossCycle_ABA + lossCycle_BAB), 'lossD': (lossDA + lossDB)}, 
					images={'realA': realA, 'realB': realB, 'fakeA': fakeA, 'fakeB': fakeB})

		#Update learning rates
		lr_scheduler_GAN.step()
		lr_scheduler_Dis_A.step()
		lr_scheduler_Dis_B.step()

		# Save Models checkpoints
		torch.save(GEN_AtoB.state_dict(), options["outputpath"]+'GEN_AtoB.pth')
		torch.save(DIS_B.state_dict(), options["outputpath"]+'DIS_B.pth')
		torch.save(GEN_BtoA.state_dict(), options["outputpath"]+'GEN_BtoA.pth')
		torch.save(DIS_A.state_dict(), options["outputpath"]+'DIS_A.pth')
		with open("EpochVerify.txt",'w') as ff:
			ff.write("\n"+str(epoch))



if __name__=="__main__":
	print("Starting the training")
	trainer(options)