# CAP5415-KeypointDetection

## Set up
First you need to download the validation dataset for 2014. There are two ways you can do this. 
1. Visit the COCO website and download http://images.cocodataset.org/zips/val2014.zip and http://images.cocodataset.org/annotations/annotations_trainval2014.zip 
2. In the command line: 
```bash
mkdir dataset
mkdir dataset/COCO/
cd dataset/COCO/
git clone https://github.com/pdollar/coco.git
cd ../../

mkdir dataset/COCO/images

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip

unzip person_keypoints_trainval2014.zip -d dataset/COCO/
unzip val2014.zip -d dataset/COCO/images

rm -f person_keypoints_trainval2014.zip
rm -f val2014.zip
```

You will also need to download the weight paths locally due to their size:
The classifier can be downloaded by the following link and should be stored in path 'classifier_utils/'
* https://drive.google.com/file/d/1xJP3o0gdHl3G2CfrfD5a68_bgUaz_PIZ/view?usp=sharing

The multipose-model weights can be downloaded by the following link and should be stored in path 'multipose_utils/multipose_model/'
* https://drive.google.com/file/d/1xJP3o0gdHl3G2CfrfD5a68_bgUaz_PIZ/view?usp=sharing

## Run
Make the respeective changes to the passed filepaths based on your directory structure and then run in Python3+ evaluation.py. 
```python
main_dir = '/home/CAP5415-KeypointDetection'
model_path = main_dir + '/coco_pose_iter_440000.pth.tar'

image_dir = main_dir + '/dataset/COCO/images'
model_path = main_dir + '/coco_pose_iter_440000.pth.tar'
output_dir = '/results'
anno_path = main_dir + '/dataset/COCO/'
vis_dir = main_dir + '/dataset/COCO/vis'
```
The scripts processes ~1000 images set aside for validation by the author of the original paper. The results will be output in the command line. 
