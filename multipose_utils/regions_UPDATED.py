import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as pyplot
import cv2
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image	

joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]

BUFFER_VERT = 10
BUFFER_HORIZ = 5
modelFileName = 'C:/Users/brand/OneDrive/UCF/Fall_2018/Computer_Vision/Project/CAP5415-KeypointDetection/evaluate/model_best.pth.tar'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def find_regions(img_orig, joint_list, person_to_joint_assoc):
	""" Find regions of potential humans
		by fisding the the max and min points with respect 
		to the x and y direction for each delcareed human then 
		produce the region of the image associated with those points """


	#For Each person
	for person_joint_info in person_to_joint_assoc:
		#For Each Limb
		print(img_orig.shape)
		#Initalization
		maxX = 0
		minX = img_orig.shape[1]
		maxY = 0
		minY = img_orig.shape[0]
		
		print(maxX)
		print(minX)
		print(maxY)
		print(minY)
		#wait = input("Enter")
		for limb_type in range(19):
			#The Indidieces of this joint
			joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
			joint_coords = joint_list[joint_indices, 0:2]
			
			if -1 in joint_indices:
				#Only consider connected limbs
				continue

			for joint in joint_coords:
				maxX = int(max(maxX , joint[0]))
				minX = int(min(minX , joint[0]))
				maxY = int(max(maxY , joint[1]))
				minY = int(min(minY , joint[1]))
		
		#Put a buffer around the potential humans
		maxX = maxX + BUFFER_HORIZ
		minX = minX - BUFFER_HORIZ
		maxY = maxY + BUFFER_VERT
		minY = minY - BUFFER_VERT
		
		if maxX > img_orig.shape[1]:
			maxX = img_orig.shape[1]		
		if minX < 0:
			minX = 0
		if maxY > img_orig.shape[0]:
			maxY = img_orig.shape[0]
		if minY < 0:
			minY = 0
		
		if maxX == minX:
			maxX = maxX + 1
		if maxY == minY:
			maxY = maxY + 1
			
		print(maxX)
		print(minX)
		print(maxY)
		print(minY)
		
		thisRegion = img_orig[minY:maxY,minX:maxX,:]
		
		model = models.__dict__['resnet152'](pretrained=True)
		model.fc = nn.Linear(2048, 2)
		model = torch.nn.DataParallel(model).cuda()
		checkpoint = torch.load(modelFileName)
		model.load_state_dict(checkpoint['state_dict'])
		model.eval()
		
		preprocess =  transforms.Compose([
#                                 transforms.Lambda(shear),
                                transforms.Scale(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
		
		thisRegionTensor = preprocess(Image.fromarray(thisRegion))
		thisRegionTensor.unsqueeze_(0)
		
		thisRegionVar = torch.autograd.Variable(thisRegionTensor)
		y_pred = model(thisRegionVar)
		smax = nn.Softmax()
		smax_out = smax(y_pred)[0]
		
		print(smax_out)
		
		cv2.imshow('test0', img_orig)
		cv2.imshow('test', thisRegion)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		
		wait = input("Enter")
			
		