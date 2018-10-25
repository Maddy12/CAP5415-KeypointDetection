import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import skimage.io as io
import pylab


main_dir = 'C:\\Users\\maddy\\OneDrive - Knights - University of Central Florida\\CAP5415-KeypointDetection\\'
ann_path = main_dir + "dataset\\COCO\\annotations\\person_keypoints_val2017.json"
json_path = main_dir + "dataset\\COCO\\test2015.json"
mask_dir = main_dir + 'dataset\\COCO\\mat'
filelist_path = main_dir + 'dataset\\COCO\\filelist.txt'
masklist_path = main_dir + 'dataset\\COCO\\masklist.txt'

annType = 'keypoints'
prefix = 'person_keypoints'

print('Running demo for {} results. '.format(annType))

dataDir = main_dir + '\\dataset\\COCO'
dataType = 'val2014'  # do not have annotations for test set unfortunately

annFile = '{}/annotations/{}_{}.json'.format(dataDir, prefix, dataType)

cocoGt = COCO(annFile)

# initialize COCO detections api
resFile = '{}/results/{}_{}_fake{}100_results.json'.format(dataDir, prefix, dataType, annType)
cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
