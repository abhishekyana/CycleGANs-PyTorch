import torch
import argparse
import itertools
from PIL import Image
from datasets import AnBDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Generator, Discriminator
from utils import ReplayBuffer, Logger, weights_init_normal

from configure import options #Edit this file to control the parameters once, without everytime pasting huge run command for every run.


inH, inW, inC = options["inResolution"]
outH, outW, outC = options["outResolution"]

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


def CycleGANmapper(inC, outC, options, init_weights=True):
	GA2B = Generator(inC, outC)
	DB = Discriminator(outC)
	GB2A = Generator(outC, inC)
	DA = Discriminator(inC)
	if options["cuda"]:
		GA2B.cuda()
		DB.cuda()
		GB2A.cuda()
		DA.cuda()
	if init_weights:
		GA2B.apply(weights_init_normal)
		DB.apply(weights_init_normal)
		GB2A.apply(weights_init_normal)
		DA.apply(weights_init_normal)
		return (GA2B, DB), (GB2A, DA)
	if options["continued"]:
		GA2B.load_state_dict(torch.load("output/GEN_AtoB.pth"))
		DB.load_state_dict(torch.load("output/DIS_B.pth"))
		GB2A.load_state_dict(torch.load("output/GEN_BtoA.pth"))
		DA.load_state_dict(torch.load("output/DIS_A.pth"))
		return (GA2B, DB), (GB2A, DA)

def SaveModels():
	torch.save(GEN_AtoB.state_dict(), options["outputpath"]+'GEN_AtoB.pth')
	torch.save(DIS_B.state_dict(), options["outputpath"]+'DIS_B.pth')
	torch.save(GEN_BtoA.state_dict(), options["outputpath"]+'GEN_BtoA.pth')
	torch.save(DIS_A.state_dict(), options["outputpath"]+'DIS_A.pth')
	print(f"Models Saved at {options['outputpath']}")

def trainer(options):
	startEpoch = options["epoch"]
	nEpochs = options["nEpochs"]
	decayEpoch = options["decayEpoch"]
	assert(nEpochs>decayEpoch), "The decay epoch is larger than total epochs, There will be no decay :P, Sure?"

	(GEN_AtoB, DIS_B), (GEN_BtoA, DIS_A) = CycleGANmapper(inC, outC, options)

	# LOSSES 
	GANLoss = torch.nn.MSELoss() # ImageA to ImageB distance
	CycleLoss = torch.nn.L1Loss() # ImageA->ImageB->ImageA' distance
	IdentityLoss  = torch.nn.L1Loss() # Absolute loss

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

			# Generator GEN_AtoB mapping from A to B and GEN_BtoA mapping from B to A
			optim_GAN.zero_grad()

			#Identity Loss: GEN_AtoB(realB) should generate B
			synthB = GEN_AtoB(realB)
			lossIden_BtoB = IdentityLoss(synthB, realB)*5.0

			#Identity Loss: GEN_BtoA(realA) should generate A
			synthA = GEN_BtoA(realA)
			lossIden_AtoA = IdentityLoss(synthA, realA)*5.0

			#GAN Loss: DIS_B(GEN_AtoB(realA)) should be closest to real target.
			fakeB = GEN_AtoB(realA)
			classB = DIS_B(fakeB)
			lossGEN_A2B = GANLoss(classB, targetReal)

			#GAN Loss: DIS_A(GEN_BtoA(realB)) should be closest to real target.
			fakeA = GEN_BtoA(realB)
			classA = DIS_A(fakeA)
			lossGEN_B2A = GANLoss(classA, targetReal)

			#Cycle Recontruction: GEN_BtoA(GEN_AtoB(realA)) -> realA should give realA
			reconA = GEN_BtoA(fakeB)
			lossCycle_ABA = CycleLoss(reconA, realA)*10.0

			#Cycle Recontruction: GEN_AtoB(GEN_BtoA(realB)) -> realB should give realA
			reconB = GEN_BtoA(fakeA)
			lossCycle_BAB = CycleLoss(reconB, realB)*10.0

			# Total Loss of the GANSs, Cycle Consistancy Loss
			lossTotal = lossCycle_BAB + lossCycle_ABA + lossGEN_B2A + lossGEN_A2B + lossIden_AtoA + lossIden_BtoB
			lossTotal.backward()

			optim_GAN.step()

			# Discriminator A Updatation part
			optim_DIS_A.zero_grad()

			#Real Loss: When a realA is sent into the Discriminator should predict 1.0
			classAreal = DIS_A(realA)
			lossDreal = GANLoss(classAreal, targetReal)

			#Fake Loss: When a fakeA is sent into the Discriminator should predict 0.0
			fakeA = FakeAHolder.push_and_pop(fakeA) # For logging
			classAfake = DIS_A(fakeA.detach())
			lossDfake = GANLoss(classAfake, targetFake)

			# The Discriminator should perfrom equally good at both the tasks
			lossDA = (lossDreal + lossDfake)/2

			lossDA.backward()
			optim_DIS_A.step()

			# Discriminator B Updatation part
			optim_DIS_B.zero_grad()

			#Real Loss: When a realB is sent into the Discriminator should predict 1.0
			classBreal = DIS_B(realB)
			lossDreal = GANLoss(classBreal, targetReal)

			#Fake Loss: When a fakeB is sent into the Discriminator should predict 0.0
			fakeB = FakeBHolder.push_and_pop(fakeB) # For logging
			classBfake = DIS_B(fakeB.detach())
			lossDfake = GANLoss(classBfake, targetFake)

			# The Discriminator should perfrom equally good at both the tasks
			lossDB = (lossDreal + lossDfake)/2

			lossDB.backward()
			optim_DIS_B.step()
			#Progress
			logger.log({'lossTotal': lossTotal, 'lossIdentity': (lossIden_AtoA + lossIden_BtoB), 'lossGAN': (lossGEN_A2B + lossGEN_B2A),
					'lossCycle': (lossCycle_ABA + lossCycle_BAB), 'lossD': (lossDA + lossDB)}, 
					images={'realA': realA, 'realB': realB, 'fakeA': fakeA, 'fakeB': fakeB})

		#Update learning rates
		lr_scheduler_GAN.step()
		lr_scheduler_Dis_A.step()
		lr_scheduler_Dis_B.step()

		# Save Models checkpoints
		SaveModels(GEN_AtoB, DIS_B, GEN_BtoA, DIS_A)
		with open("EpochVerify.txt",'w') as ff:
			ff.write("\n"+str(epoch))



if __name__=="__main__":
	trainer(options)
