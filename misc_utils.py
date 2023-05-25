import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from tqdm import tqdm
import pickle
from scipy import signal
import pandas as pd
import re # Remove duplicate string
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sys.path.append('ritm_interactive_segmentation')
# The package from RITM https://github.com/yzluka/ritm_interactive_segmentation
from isegm.inference.clicker import Click
from isegm.inference import utils as is_utils
from isegm.inference.predictors import get_predictor as is_get_predictor  
from isegm.inference.evaluation import evaluate_sample_onepass as is_evaluate_sample_onepass
from isegm.inference.evaluation import evaluate_sample_onepass_preset_image_no_iou
# Some global variables
sam_checkpoint = "pretrained_model_weight/sam_vit_h_4b8939.pth"
ritm_checkpoint = 'pretrained_model_weight/coco_lvis_h32_itermask.pth'
model_type = "vit_h"
device = "cuda"

def load_predictor(Flag_SAM_True_RITM_False):
    """
    A unified API to load the predictor for both SAM and RITM
    """
    print('...loading {} predictor'.format('SAM' if Flag_SAM_True_RITM_False else 'RITM'))
    if Flag_SAM_True_RITM_False:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    else:
        model = is_utils.load_is_model(ritm_checkpoint, "cuda")
        predictor = is_get_predictor(model, "NoBRS", "cuda")
    return predictor

def set_image_predictor(predictor, image, Flag_SAM_True_RITM_False):
    """
    A unified API to set image for the predictor
    """
    if Flag_SAM_True_RITM_False:
        predictor.set_image(image)
    else:
        predictor.set_input_image(image)
    return
    
def show_mask(mask, ax, random_color=False):    # Copied from predictor_example.ipynb
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):       # Copied from predictor_example.ipynb
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):                      # Copied from predictor_example.ipynb
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


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

def get_most_inner_point_from_a_binary_map(input_mask):
    """
    The function that finds the most inner point of a binary map
    The most "inner point" is defined as the furthest distance to the 0 (background) pixel of target value-1 pixels
    Ref from RITM https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
    and https://github.com/mazurowski-lab/segment-anything-medical-evaluation/blob/main/prompt_gen_and_exec_v1.py#LL339C17-L339C17
    """
    input_mask = np.uint8(input_mask)
    # First lets pad the input mask, so that the edge can get accounted for
    padded_mask = np.pad(input_mask, ((1, 1), (1, 1)), 'constant')
    # Now call the CV2 package for calculating the distance with distance type L2, also only take the original (remove padding)
    dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
    # Extract the centroid position
    cY, cX = np.where(dist_img==dist_img.max())
    return cY, cX

def choose_mask(masks, scores, ground_truth, oracle=False):
    """
    The function that choses the mask the return the index of such mask
    :param masks: 3 masks returned by SAM
    :param scores: 3 scores returned by SAM
    :param ground_truth: The GT masks of such object
    :param oracle: A binary flag whether this is an oracle (True) or just return the max confidence (False)
    """
    if not oracle:
        return np.argmax(scores)
    else:   # We need to select the oracle one with best IoU then
        IoU_list = get_IoU_for_3masks(ground_truth, masks)
        return np.argmax(IoU_list)

def prompt_with_point(predictor, input_point, save_mask_path, save_mask_prefix, 
                    input_label = np.array([1])):
    """
    Prompt the predictor with img and point pair and save the mask in the save_mask_name
    :param predictor: The SAM loaded predictor, which has already been set_image before
    :param img_path: The location of the image
    :param input_point: A single point of prompting
    :param save_mask_prefix: The prefix to save the mask
    :param save_mask_path: The position to save the mask
    :param input_label: This is for the signifying whether it is foreground or background
    """
    # Make inference using SAM
    masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,)
    # make the save path
    save_name = os.path.join(save_mask_path, 
                             save_mask_prefix + '_{}_{}_{}.npy'.format(scores[0],scores[1],scores[2]))
    np.save(save_name, masks)

def prompt_with_multiple_points(predictor, input_points, input_labels, 
                                save_mask_path, save_mask_prefix, png_mode=True, 
                                logit_refine=None): # Pickle mode is not saving space
    """
    Prompt the predictor with img and point pair and save the mask in the save_mask_name
    :param predictor: The SAM loaded predictor, which has already been set_image before
    :param img_path: The location of the image
    :param input_point: A single point of prompting
    :param save_mask_prefix: The prefix to save the mask
    :param save_mask_path: The position to save the mask
    :param input_label: This is for the signifying whether it is foreground or background
    """
    # Make sure they are all numpy arrays
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    # Make inference using SAM
    masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=logit_refine,
            multimask_output=True,)
    
    if png_mode:
        save_name = os.path.join(save_mask_path, 
                                save_mask_prefix + '_{}_{}_{}.png'.format(scores[0],scores[1],scores[2]))
        cv2.imwrite(save_name, np.swapaxes(masks, 0, 2)*255)
    else:
        # make the save path
        save_name = os.path.join(save_mask_path, 
                                save_mask_prefix + '_{}_{}_{}.npy'.format(scores[0],scores[1],scores[2]))
        np.save(save_name, masks)

    return masks, scores, logits

def prompt_RITM_with_multiple_points(predictor, input_point_list, image, 
                                     gt_mask, save_mask_path, save_mask_prefix):
    """
    Calling the RITM inference code API and then saving the results down
    """
    preds, _ = evaluate_sample_onepass_preset_image_no_iou(
                                            gt_mask=gt_mask,
                                            predictor=predictor, 
                                            clicks=input_point_list)
    # Saving the RITM output
    save_name = os.path.join(save_mask_path, save_mask_prefix + '.png')
    preds = preds[:,:,None].repeat(3,2)
    cv2.imwrite(save_name, preds*255)
    return preds

def prompt_with_mask(predictor, input_mask, save_mask_path, save_mask_prefix, ):
    """
    The function to prompt the SAM model with a mask of the same image
    """
    # Make inference using SAM
    masks, scores, logits = predictor.predict(
            mask_input = input_mask,
            multimask_output=True,)
    # make the save path
    save_name = os.path.join(save_mask_path, 
                             save_mask_prefix + '_{}_{}_{}.png'.format(scores[0],scores[1],scores[2]))
    cv2.imwrite(save_name, np.swapaxes(masks, 0, 2)*255)

def prompt_with_bbox(predictor, input_bbox, save_mask_path, save_mask_prefix,):
    masks, _, _ = predictor.predict(
            box=input_bbox[None, :],   # This is not sure
            multimask_output=False,)
    # make the save path
    save_name = os.path.join(save_mask_path, 
                             save_mask_prefix + '.png')
    # print(np.shape(masks))
    cv2.imwrite(save_name, masks.astype(np.uint8)[0, :, :]*255)
    # np.save(save_name, masks)
