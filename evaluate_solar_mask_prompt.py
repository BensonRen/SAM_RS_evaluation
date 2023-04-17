import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
from show_img import *
import glob
from multiprocessing import Pool



def IoU_single_object_mask(gt_mask, pred_mask):
    """
    gt_mask and pred_mask are both 0/1 valued masks
    """
    intersection = gt_mask * pred_mask
    union = (gt_mask + pred_mask) > 0
    num_intersection = np.sum(intersection)
    num_union = np.sum(union)
    return  num_intersection / num_union, num_intersection, num_union

def get_IoU_for_3masks(gt_mask, pred_3masks):
    IoU_list = np.zeros(3)
    for i in range(3):
        IoU_list[i], _ , _ = IoU_single_object_mask(gt_mask, pred_3masks[i, :, :])
    return IoU_list

def evaluate_pixel_IoU_folder_of_mask(mask_folder, gt_folder='solar_masks', 
                                      debug_plot=False, num_limit=999999):
    """
    The function that evaluates the best pixel IOU taking the first finetune map
    """
    save_df = pd.DataFrame(columns=['img_name', 'num_intersection', 'num_union'])
    # num_intersection_tot, num_union_tot = 0, 0
    for mask in tqdm(os.listdir(mask_folder)):
        if '.tif' not in mask:
            continue
        cur_mask = cv2.imread(os.path.join(mask_folder, mask))[:, :, 0] > 0
        gt_mask_file = os.path.join(gt_folder, mask)
        if not os.path.exists(gt_mask_file):
            print('file {} made prediction while no such gt'.format(gt_mask_file))
            # num_union_tot += np.sum(cur_mask)
            save_df.loc[len(save_df)] = [mask, 0, np.sum(cur_mask)]
            continue
        gt_mask = cv2.imread(gt_mask_file)[:, :, 0] > 0
        _, num_intersection, num_union = IoU_single_object_mask(gt_mask, cur_mask)
        save_df.loc[len(save_df)] = [mask, num_intersection, num_union]
        if debug_plot:
            plot_img_gt_mask_and_detector_mask(mask)

        num_limit -= 1
        if num_limit <= 0:
            return save_df
        
        # num_intersection_tot += num_intersection
        # num_union_tot += num_union
    return save_df

def evaluated_prompted_mask(mask_prompt_folder, SAM_mask_fodlder, 
                            gt_folder='solar_masks',num_limit=999999,
                            plot_compare=False):
    """
    The function that evaluates the SAM improved mask from detector mask
    Note that we are taking the first level of the output which is the most local level of the output
    for the "fine tune" purpose of the original map
    """
    save_df = pd.DataFrame(columns=['img_name', 'm0_num_intersection','m0_num_union','m1_num_intersection','m1_num_union','m2_num_intersection','m2_num_union'])
    # num_intersection_tot, num_union_tot = 0, 0
    for mask in tqdm(os.listdir(mask_prompt_folder)):
        if '.tif' not in mask:
            continue

        gt_mask_file = os.path.join(gt_folder, mask)
        if not os.path.exists(gt_mask_file):
            print('file {} made prediction while no such gt'.format(gt_mask_file))
            # num_union_tot += np.sum(cur_mask)
            # if agg_SAM_mask is None:
            #     union = 0
            # else:
            #     union = np.sum(agg_SAM_mask)
            # save_df.loc[len(save_df)] = [mask, 0, union]
            continue
        gt_mask = cv2.imread(gt_mask_file)[:, :, 0] > 0
        # cur_mask = cv2.imread(os.path.join(mask_prompt_folder, mask))#[:, :, 0]# > 0
        mask_file_reg_exp = '{}_prompt_ind_*.png'.format(mask.replace('.tif',''))
        mask_file_list = glob.glob(os.path.join(SAM_mask_fodlder, mask_file_reg_exp))
        agg_SAM_mask = None
        print(len(mask_file_list))
        for SAM_mask_file in mask_file_list:
            cur_mask = cv2.imread(SAM_mask_file) > 0
            # cur_mask = np.swapaxes(cur_mask, 0, 2) > 0 
            # cur_mask = cur_mask#[0, :, :]
            if agg_SAM_mask is None:
                agg_SAM_mask = cur_mask
            else:
                agg_SAM_mask += cur_mask # Make a union of all the prediction
        if agg_SAM_mask is None:
            agg_SAM_mask = np.expand_dims(np.zeros_like(gt_mask), 0)
        else:
            agg_SAM_mask = np.swapaxes(agg_SAM_mask, 0, 2) > 0     # Threshold to make it binary
            if plot_compare:
                plot_img_gt_mask_and_detector_mask_and_all_three_SAM_output_channel(mask, agg_SAM_mask)
        
        append_list = [mask]
        for i in range(3):
            _, num_intersection, num_union = IoU_single_object_mask(gt_mask, agg_SAM_mask[min(len(agg_SAM_mask) - 1, i), :, :])
            append_list.extend([num_intersection, num_union])
        save_df.loc[len(save_df)] = append_list

        num_limit -= 1
        if num_limit <= 0:
            return save_df
        
        # num_intersection_tot += num_intersection
        # num_union_tot += num_union
    return save_df

def plot_img_gt_mask_and_detector_mask(img_name,
                                       SAM_mask=None,
                                       save_folder='investigation/solar_pv_detector',
                                       img_folder='solar-pv', 
                                       gt_folder='solar_masks', 
                                       detector_mask_folder='solar_finetune_mask'):
    img = cv2.imread(os.path.join(img_folder, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # No need to change cuz this is self saved
    gt_mask = cv2.imread(os.path.join(gt_folder, img_name))[:, :, 0]
    detector_mask = cv2.imread(os.path.join(detector_mask_folder, img_name))[:, :, 0]
    f = plt.figure(figsize=[15,5])
    ax1 = plt.subplot(141)
    ax1.imshow(img)
    plt.axis('off')
    ax2 = plt.subplot(142)
    ax2.imshow(gt_mask*255)
    plt.axis('off')
    ax2 = plt.subplot(143)
    ax2.imshow(detector_mask)
    plt.axis('off')
    if SAM_mask is not None:
        ax2 = plt.subplot(144)
        ax2.imshow(SAM_mask*255)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, img_name))

def plot_img_gt_mask_and_detector_mask_and_all_three_SAM_output_channel(img_name,
                                       SAM_mask,
                                       save_folder='investigation/solar_pv_detector',
                                       img_folder='solar-pv', 
                                       gt_folder='solar_masks', 
                                       detector_mask_folder='solar_finetune_mask'):
    img = cv2.imread(os.path.join(img_folder, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # No need to change cuz this is self saved
    gt_mask = cv2.imread(os.path.join(gt_folder, img_name))[:, :, 0]
    detector_mask = cv2.imread(os.path.join(detector_mask_folder, img_name))[:, :, 0]
    f = plt.figure(figsize=[15,10])
    plt.tight_layout()
    ax1 = plt.subplot(231)
    ax1.imshow(img)
    plt.axis('off')
    ax2 = plt.subplot(232)
    ax2.imshow(gt_mask*255)
    plt.axis('off')
    ax2 = plt.subplot(233)
    ax2.imshow(detector_mask)
    plt.axis('off')
    for i in range(3):
        ax2 = plt.subplot(int('23{}'.format(i+4)))
        ax2.imshow(SAM_mask[i, :, :]*255)
        plt.axis('off')
    plt.savefig(os.path.join(save_folder, img_name))

# def plot_img_gt_mask_and_detector_mask_and_all_three_SAM_output_channel_and_prompt_map(img_name,
#                                        SAM_mask,
#                                        save_folder='investigation/solar_pv_detector',
#                                        img_folder='solar-pv', 
#                                        gt_folder='solar_masks', 
#                                        detector_mask_folder='solar_finetune_mask'):
#     img = cv2.imread(os.path.join(img_folder, img_name))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # No need to change cuz this is self saved
#     gt_mask = cv2.imread(os.path.join(gt_folder, img_name))[:, :, 0]
#     detector_mask = cv2.imread(os.path.join(detector_mask_folder, img_name))[:, :, 0]
#     f = plt.figure(figsize=[15,5])
#     ax1 = plt.subplot(231)
#     ax1.imshow(img)
#     plt.axis('off')
#     ax2 = plt.subplot(232)
#     ax2.imshow(gt_mask*255)
#     plt.axis('off')
#     ax2 = plt.subplot(233)
#     ax2.imshow(detector_mask)
#     plt.axis('off')
#     for i in range(3):
#         ax2 = plt.subplot(int('23{}'.format(i+4)))
#         ax2.imshow(SAM_mask[i, :, :]*255)
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_folder, img_name))

if __name__ == '__main__':
    print('running evaluate solar mask prompt...')
    # Calculate the pixel IoU count of the original masks provided by the detector model
    # save_df = evaluate_pixel_IoU_folder_of_mask(mask_folder='solar_finetune_mask')#, num_limit=20)
    # save_df.to_csv('detector_pixel_iou.csv')
    # print('average pixel IoU of trained detector is {}%, \
    # ({} intersection pixels, {} union pixels)'.format(num_intersection_tot/num_union_tot*100,num_intersection_tot,num_union_tot))
    
    # Pixel IoU for the SAM adjusted mask
    # save_df = evaluated_prompted_mask(mask_prompt_folder='solar_finetune_mask', 
    #                                   SAM_mask_fodlder='solar_pv_mask_step_20_prompt_save',
    #                                   plot_compare=True,
    #                                    num_limit=30)
    # save_df.to_csv('detector_pixel_iou.csv')
    