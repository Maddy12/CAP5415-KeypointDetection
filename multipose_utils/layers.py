import torch
import torch.nn as nn
from multipose_utils.generate_pose import *


class RegionProposal():
    def __init__(self, output1, output2):
        """
        To get heatmaps and pafs:
            heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
            pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
        :param output1: Multipose model output 1
        :param output2: Multipose model output 2
        """
        self.paf = output1
        self.heatmaps = output2  # heatmap
        self.param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}

    def forward(self):
        #TODO

    def NMS(self, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False):
        """
        From https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
        NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
        :param heatmaps: set of grayscale images on which to find local maxima (3d np.array,
        with dimensions image_height x image_width x num_heatmaps)
        :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
            Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
        :param bool_refine_center: Flag indicating whether:
            - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
            - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
        fine-tune the location of the peak at the resolution of the original input image
        :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
        to each upsampled patch before fine-tuning the location of each peak.
        :return: a NUM_JOINTS x 4 np.array where each row represents a joint type (0=nose, 1=neck...)
        and the columns indicate the {x,y} position, the score (probability) and a unique id (counter)
        """
        joint_list_per_joint_type = []
        cnt_total_joints = 0

        # For every peak found, win_size specifies how many pixels in each
        # direction from the peak we take to obtain the patch that will be
        # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
        # (for BICUBIC interpolation to be accurate, win_size needs to be >=2!)
        win_size = 2

        for joint in range(NUM_JOINTS):
            map_orig = self.heatmaps[:, :, joint]
            peak_coords = find_peaks(self.param, map_orig)
            peaks = np.zeros((len(peak_coords), 4))
            for i, peak in enumerate(peak_coords):
                if bool_refine_center:
                    x_min, y_min = np.maximum(0, peak - win_size)
                    x_max, y_max = np.minimum(
                        np.array(map_orig.T.shape) - 1, peak + win_size)

                    # Take a small patch around each peak and only upsample that
                    # tiny region
                    patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                    map_upsamp = cv2.resize(
                        patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                    # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                    # more than one peak per joint!) -> For now, skip it (it's
                    # accurate enough)
                    map_upsamp = gaussian_filter(
                        map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                    # Obtain the coordinates of the maximum value in the patch
                    location_of_max = np.unravel_index(
                        map_upsamp.argmax(), map_upsamp.shape)
                    # Remember that peaks indicates [x,y] -> need to reverse it for
                    # [y,x]
                    location_of_patch_center = compute_resized_coords(
                        peak[::-1] - [y_min, x_min], upsampFactor)
                    # Calculate the offset wrt to the patch center where the actual
                    # maximum is
                    refined_center = (location_of_max - location_of_patch_center)
                    peak_score = map_upsamp[location_of_max]
                else:
                    refined_center = [0, 0]
                    # Flip peak coordinates since they are [x,y] instead of [y,x]
                    peak_score = map_orig[tuple(peak[::-1])]
                peaks[i, :] = tuple([int(round(x)) for x in compute_resized_coords(
                    peak_coords[i], upsampFactor) + refined_center[::-1]]) + (peak_score, cnt_total_joints)
                cnt_total_joints += 1
            joint_list_per_joint_type.append(peaks)

        return joint_list_per_joint_type
