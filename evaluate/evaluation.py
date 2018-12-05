import sys
sys.path.append('..')

# Local
from evaluate.coco_eval import *
from multipose_utils.post import get_persons
from multipose_utils.multipose_model import get_multipose_model
from classifier_utils import classifier_model
from multipose_utils.regions import find_regions


def run_eval(multipose, classifier, image_dir, anno_dir, vis_dir, image_list_txt, preprocess):
    """Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    # This txt file is fount in the caffe_rtpose repository:
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master
    img_ids, img_paths, img_heights, img_widths = get_coco_val(
        image_list_txt)
    print("Total number of validation images {}".format(len(img_ids)))
    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    for i in range(len(img_ids)):
        if i % 10 == 0 and i != 0:
            print("Processed {} images".format(i))

        oriImg = cv2.imread(os.path.join(image_dir, 'val2014/' + img_paths[i]))
        shape_dst = np.min(oriImg.shape[0:2])

        multiplier = get_multiplier(oriImg)
        orig_paf, orig_heat = get_outputs(
            multiplier, oriImg, multipose, preprocess)

        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, multipose, preprocess)

        paf, heatmap = handle_paf_and_heat(
            orig_heat, flipped_heat, orig_paf, flipped_paf)
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        person_to_joint_assoc, joint_list = get_persons(oriImg, param, heatmap, paf)
        y_preds = find_regions(classifier, oriImg, joint_list, person_to_joint_assoc)
        preds = np.argmax(y_preds.cpu().detach().numpy(), axis=1)
        idx = np.argwhere(np.asarray(preds))
        try:
            filtered = person_to_joint_assoc[idx[0]]
            filtered = filtered.reshape(filtered.shape[0], filtered.shape[-1])
            append_result(img_ids[i], filtered, joint_list, outputs)
        except Exception as e:
            error = e
            import pdb;
            pdb.set_trace()
        append_result(img_ids[i], filtered, joint_list, outputs)
    eval_coco(outputs=outputs, dataDir=anno_dir, imgIds=img_ids)


if __name__ == '__main__':
    main_dir = '/home/CAP5415-KeypointDetection/'
    image_dir = os.path.join(main_dir, 'dataset/COCO_data/images')
    model_path = os.path.join(main_dir, 'multipose_utils/multipose_model/coco_pose_iter_440000.pth.tar')
    output_dir = os.path.join(main_dir, 'results')
    anno_dir = os.path.join(main_dir, 'dataset/COCO_data/annotations')
    vis_dir = os.path.join(main_dir, 'dataset/COCO_data/vis')
    preprocess = 'rtpose'
    post_model_path = os.path.join(main_dir, 'classifier_utils/model_best.pth.tar')
    image_list_txt = os.path.join(main_dir, 'evaluate/image_info_val2014_1k.txt')

    # Init Models
    multipose = get_multipose_model(model_path)
    classifier = classifier_model.get_model(post_model_path)
    run_eval(multipose, classifier, image_dir, anno_dir, vis_dir, image_list_txt, preprocess)




