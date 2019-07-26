options = {
			"epoch": 0,
			"nEpochs": 200,
			"batchsize": 2,
			"datapath": "./young2old",
			"outputpath":'./output/',
			"learningrate": 2e-4,
			"decayEpoch": 50,
			"inResolution":[256, 256, 3],
			"outResolution":[256, 256, 3],
			"continued": False,
			"nThreads":10,
			"unpaired":True,
		   }

testoptions = {
				"batchsize": 1,
				"datapath": './young2old',
				"outputpath":'./output/',
				"inResolution":[256, 256, 3],
				"outResolution":[256, 256, 3],
				"nThreads":10,
				"GEN_AtoB":'./output/GEN_AtoB.pth',
				"GEN_BtoA":'./output/GEN_BtoA.pth'
			  }