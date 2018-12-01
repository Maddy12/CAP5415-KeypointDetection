import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as pyplot
import cv2
joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]

def find_regions(img_orig, joint_list, person_to_joint_assoc):
	#For Each person
	for person_joint_info in person_to_joint_assoc:
		#For Each Limb
		print(img_orig.shape)
		maxX = 0
		minX = img_orig.shape[1]
		maxY = 0
		minY = img_orig.shape[0]
		
		print(maxX)
		print(minX)
		print(maxY)
		print(minY)
		wait = input("Enter")
		for limb_type in range(19):
			#The Indidieces of this joint
			joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
			joint_coords = joint_list[joint_indices, 0:2]

			for joint in joint_coords:
				maxX = int(max(maxX , joint[0]))
				minX = int(min(minX , joint[0]))
				maxY = int(max(maxY , joint[1]))
				minY = int(min(minY , joint[1]))
				
		print(maxX)
		print(minX)
		print(maxY)
		print(minY)
		
		thisRegion = img_orig[minY:maxY,minX:maxX,:]
		
		
		cv2.imshow('test0', img_orig)
		cv2.imshow('test', thisRegion)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		
		wait = input("Enter")
			
		