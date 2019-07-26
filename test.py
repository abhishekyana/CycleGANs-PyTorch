import os
import torch
from models import Generator
from datasets import AnBDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from configure import testoptions as options
from torchvision.utils import make_grid, save_image


inH, inW, inC = options["inResolution"]
outH, outW, outC = options["outResolution"]

if torch.cuda.is_available():
	options["cuda"]=True

def loadModel(options):
	GEN_A2B = Generator(inC, outC)
	GEN_B2A = Generator(outC, inC)
	if options["cuda"]:
		GEN_A2B.cuda()
		GEN_B2A.cuda()
	GEN_A2B.load_state_dict(torch.load(options["GEN_A2B"]))
	GEN_B2A.load_state_dict(torch.load(options["GEN_B2A"]))
	GEN_A2B.eval()
	GEN_B2A.eval()
	return GEN_A2B, GEN_B2A

def loadData(options):
	ApplyTransforms = [ transforms.ToTensor(),
                		transforms.Normalize((1/2,)*3, (1/2,)*3) ]
	dataloader = DataLoader(ImageDataset(options["datapath"], transforms_arg=ApplyTransforms, mode='test'), 
                        	batch_size=options["batchsize"], shuffle=False, num_workers=options["nThreads"])
	return dataloader

def tester(options):
	GEN_A2B, GEN_B2A = loadModel(options)
	dataloader = loadData(options)

	#Create placeholders for the images
	if options["cuda"]:
		Tensor = torch.cuda.FloatTensor
	else:
		Tensor = torch.Tensor
	imageA = Tensor(batchsize, inC, inH, inW)
	imageB = Tensor(batchsize, outC, outH, outW)

	if not os.path.exists(f'{options["outputpath"]}/A'):
	    os.makedirs(f'{options["outputpath"]}/A')
	if not os.path.exists(f'{options["outputpath"]}/B'):
	    os.makedirs(f'{options["outputpath"]}/B')
	
	for batch_id, batch_data in enumerate(dataloader):
		realA = Variable(imageA.copy_(batch_data['A']))
		realB = Variable(imageB.copy_(batch_data['B']))
		
		fakeB = GEN_A2B(realA).data
		fakeA = GEN_B2A(realB).data

		IMAGEA = torch.cat([0.5*(realA+1.0), 0.5*(fakeB+1.0)])
		IMAGEB = torch.cat([0.5*(realB+1.0), 0.5*(fakeA+1.0)])

		save_image(IMAGEA, f'{options["outputpath"]}/A/{batch_id+1}')
		save_image(IMAGEB, f'{options["outputpath"]}/B/{batch_id+1}')
		print(f"Generated image {batch_id+1} of {len(dataloader)}")
	print("DONE!! :D")

if __name__=="__main__":
	tester(options)