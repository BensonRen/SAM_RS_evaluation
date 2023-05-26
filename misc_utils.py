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

# For SpaceNet shapely operations and rastering the shapefile
from shapely.geometry.polygon import Polygon
import shapely.wkt
import rasterio.features

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

def load_predictor(Flag_SAM_True_RITM_False=True):
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
    
def get_numLabels_and_indicator_mask_list_for_SpaceNet(img_path, img_size):
    """
    Get the number of objects for current image
    """
    # First get the number of objects from a pre-saved shortcut frequency dictionary
    with open('datasets/SpaceNet6/SummaryData/SpaceNet6_obj_count_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    img_id = img_path.split('RGB_')[1].split('.')[0]
    numLabels =  loaded_dict[img_id]
    # Read the huge .csv file to get the shapely objects for current image
    shapefile = pd.read_csv('datasets/SpaceNet6/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    sub_file = shapefile.loc[shapefile['ImageId'] == img_id, :]
    assert len(sub_file) == numLabels, 'The shortcut #object does not match with the sub dataframe cut from same file name'
    # Some of the images does not conatian any buildings but still have a row with EMPTY
    if numLabels == 1 and 'EMPTY' in sub_file['PolygonWKT_Pix'].values[0]:
        return numLabels, None
    # print(sub_file)
    # print(sub_file['PolygonWKT_Pix'])
    indicator_mask_list = []    # Initialize the list
    for i in range(len(sub_file)):
        plg = shapely.wkt.loads(sub_file['PolygonWKT_Pix'].values[i])
        x, y = plg.exterior.xy
        mask = rasterio.features.rasterize([plg], out_shape=img_size)
        indicator_mask_list.append(mask)
    return numLabels, indicator_mask_list

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
            box=input_bbox[None, :],   
            multimask_output=False,)
    # make the save path
    save_name = os.path.join(save_mask_path, 
                             save_mask_prefix + '.png')
    cv2.imwrite(save_name, masks.astype(np.uint8)[0, :, :]*255)

def process_str_bbox_into_bbox(bbox_str,
                            bbox_permutation=[1,0,3,2],):
    """
    Because the bbox saved in pandas df is in format of string and out of position
    process it into desired format of a np array and permutate it into position
    """
    # First remove the [ ]
    processed_str = bbox_str.replace('[', '').replace(']','').strip()
    processed_str = re.sub(' +',' ', processed_str)
    numbers_str_list = processed_str.split(' ')
    assert len(numbers_str_list) == 4, 'Your bbox in df string is not split into 4 piece?\
          str is {} -> {}'.format(bbox_str, numbers_str_list)
    return  np.array([int(a) for a in numbers_str_list])[bbox_permutation]


################################################## 
### Legacy code for prompting with other forms ### 
##################################################

# def prompt_folder_with_point(mode, max_img=999999):
#     # Make the saving folder
#     save_mask_path = 'solar_pv_{}_prompt_save'.format(mode)
#     if not os.path.isdir(save_mask_path):
#         os.makedirs(save_mask_path)

#     # Load the points to be prompted
#     print('...loading pickel of prompt points')
#     with open('solar_pv_{}_prompt_dict.pickle'.format(mode), 'rb') as handle:
#         prompt_point_dict = pickle.load(handle)
    
#     # Load predictor
#     print('...loading predictor')
#     predictor = load_predictor()

#     # Loop over all the keys inside the prompt_point_dict
#     for img_name in tqdm(prompt_point_dict.keys()):
#         # Get image path
#         img_path = os.path.join('solar-pv', img_name)
#         # Make sure this image exist
#         if not os.path.exists(img_path):
#             print('Warning!!! {} does not exist, bypassing now'.format(img_path))
#             continue
#         # Load the image and transform color
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Why do we need to cvt color?
#         # Set the predictor
#         predictor.set_image(image)

#         # Get the input points (they are all in the form of a list)
#         for ind, input_point in enumerate(prompt_point_dict[img_name]):
#             save_mask_prefix = img_name.split('.')[0] + '_prompt_ind_{}_'.format(ind)
#             input_point_np = np.reshape(np.array(input_point), [1, 2])
#             prompt_with_point(predictor, input_point_np, save_mask_path, save_mask_prefix)
        
#         if max_img is not None:
#             max_img -= 1
#             if max_img < 0:
#                 break

# def prompt_folder_with_mask(prompt_mask_dir, mode='mask', 
#                             mask_prop_dict={'choice':'step', 'mag':50}, 
#                             mask_file_postfix='.tif', 
#                             save_prompt=False, max_img=999999):
#     """
#     The function that prompts for the folder with all the masks inside
#     """ 
#     # Make the saving folder
#     save_mask_path = 'solar_pv_{}_{}_{}_prompt_save'.format(mode, mask_prop_dict['choice'], mask_prop_dict['mag'])
#     if not os.path.isdir(save_mask_path):
#         os.makedirs(save_mask_path)
    
#     # Load predictor
#     print('...loading predictor')
#     predictor = load_predictor()

#     # Loop over all the keys inside the prompt_point_dict
#     for img_name in tqdm(os.listdir(prompt_mask_dir)):
#         # Get image path
#         img_path = os.path.join('solar-pv', img_name)
#         # Make sure this image exist
#         if not os.path.exists(img_path):
#             print('Warning!!! {} does not exist, bypassing now'.format(img_path))
#             continue
#         # Load the image and transform color
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Set the predictor
#         predictor.set_image(image)
        
#         # Process the mask (get connected component)
#         mask = cv2.imread(os.path.join(prompt_mask_dir, img_name))     # Get the single dimension mask
#         output = cv2.connectedComponentsWithStats(mask[:, :, 0], 4)
#         # Get them into structured outputs
#         (numLabels, labels, stats, centroids) = output
#         if np.max(mask) > 1:    # This is to make the mask binary
#             mask_binary = mask[:, :, 0] > 122
#         else:
#             mask_binary = mask[:, :, 0]
#         mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1

#         # Get the input points (they are all in the form of a list)
#         for i in range(numLabels):
#             prompt_mask_input = mask_mul_labels == (i+1)
#             num_pixel = np.sum(prompt_mask_input)
#             if num_pixel == 0: # Identify background and skip this
#                 continue
#             # Get the save name sorted out
#             save_mask_prefix = img_name.split('.')[0] + '_prompt_ind_{}_numPixel_{}'.format(i, num_pixel)
            
#             # Make the prompt mask
#             prompt_mask_input = prompt_mask_input.astype('float')
#             prompt_mask = make_prompt_mask(prompt_mask_input, mask_prop_dict)
#             if save_prompt:
#                 np.save(os.path.join(save_mask_path, save_mask_prefix+'_prompt_mask.npy'),prompt_mask)
#             prompt_mask = cv2.resize(prompt_mask, (256, 256))
#             prompt_mask = np.expand_dims(prompt_mask, 0)
#             # Actual prompting the SAM
#             prompt_with_mask(predictor, prompt_mask, save_mask_path, save_mask_prefix)
        
#         if max_img is not None:
#             max_img -= 1
#             if max_img < 0:
#                 break

# def make_prompt_mask(prompt_mask_input, mask_prop_dict):
#     """
#     The function to make the prompt mask from a given binary map into the logit space of SAM
#     """
#     mag = mask_prop_dict['mag']
#     background = mask_prop_dict['background']
#     if mask_prop_dict['choice'] == 'step':
#         # A series of transformations to make it [-mag, mag] with shape [1, 256, 256]
#         prompt_mask = prompt_mask_input*mag
#     elif mask_prop_dict['choice'] == 'kernel':
#         prompt_mask = prompt_mask_input*mag
#         # kernel = np.array([[0, 1, 2, 1, 0],
#         #                     [1, 2, 3, 2, 1],
#         #                     [2, 3, 4, 3, 2],
#         #                     [1, 2, 3, 2, 1],
#         #                     [0, 1, 2, 1, 0],
#         #                     ], dtype='float') # Gx + j*Gy
#         kernel = np.ones((mask_prop_dict['kernel_size'],mask_prop_dict['kernel_size']))
#         # kernel /= np.sum(kernel)/4
#         prompt_mask = signal.convolve2d(prompt_mask_input, kernel, boundary='symm', mode='same')
    
#     prompt_mask[prompt_mask == 0] = background
#     prompt_mask = cv2.resize(prompt_mask, (256, 256))    
#     return prompt_mask
    