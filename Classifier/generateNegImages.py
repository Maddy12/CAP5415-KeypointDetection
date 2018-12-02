import os
import cv2

#dir = "D:/INRIAPerson/Test/neg/"
dir = "D:/INRIAPerson/Train/neg/"
#outputdir = "D:/INRIAPerson/70X134H96/Test/neg/"
outputdir = "D:/INRIAPerson/96X160H96/Train/neg/"

i = 1
for filename in os.listdir(dir):
	img = cv2.imread(dir + filename)
	croppedImg = img[0:160, 0:96, :]
	
	cv2.imwrite(outputdir + filename, croppedImg)
	
	print(i)
	i = i+1

	