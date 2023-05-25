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
from misc_utils import *

def prompt_folder_with_multiple_points(mode, num_point_prompt, max_img=999999, 
                                       dataset='solar_pv',
                                       mask_folder='datasets/solar_masks',
                                       img_folder='datasets/solar-pv',
                                       img_postfix='tif',
                                       mask_postfix='tif',
                                       size_limit=0,
                                       SAM_refine_feedback=True,
                                       choose_oracle=False):
    # Set up the saving place
    save_mask_path = '{}_{}_prompt_save_numpoint_{}_oracle_{}'.format(dataset, 
                                                                      mode, 
                                                                      num_point_prompt,
                                                                      choose_oracle)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    img_name_list = [file for file in os.listdir(mask_folder) if mask_postfix in file]
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()

    # Pre-allocate a dictionary to save the prompt point information
    prompt_point_dict = {}

    # Loop over all the keys inside the prompt_point_dict
    for img_name in tqdm(img_name_list):
        cur_img_prompt_point = {}
        # Get image path
        img_path = os.path.join(img_folder, img_name.replace(mask_postfix, img_postfix))
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue
        # Load the image and transform color
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Set the predictor
        predictor.set_image(image)
        # Get the masks and connected components
        mask = cv2.imread(os.path.join(mask_folder, img_name))
        # Find the connected component
        output = cv2.connectedComponentsWithStats(
                mask[:, :, 0], 4)
        # Get them into structured outputs
        (numLabels, labels, stats, centroids) = output

        if np.max(mask > 1):    # This is to make the mask binary
            mask_binary = mask[:,:,0] > 122
        else:
            mask_binary = mask[:, :, 0]
        mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1

        # Loop over the each "objects" in this current image
        for i in range(numLabels):
            cur_img_prompt_point[i] = {}
            num_pixel = np.sum(mask_mul_labels == (i+1))
            # First identify background
            if num_pixel <= size_limit:
                continue
            indicator_mask = mask_mul_labels == (i + 1)

            # Set up the point list and the label list
            input_point_list, input_label_list = [], []
            last_logit = None   # Initialize the last logit to be None

            for j in range(num_point_prompt):
                save_mask_prefix = img_name.split('.')[0] + '_prompt_ind_{}_point_num_{}'.format(i, j)
                if j == 0:  # The first prompt is special, because it does not have 
                    cY, cX = get_most_inner_point_from_a_binary_map(indicator_mask)
                else: # The iterative experience where each time get a point from 
                    error_mask = np.uint8(np.bitwise_xor(indicator_mask, 
                                                         last_prediction_mask))
                    cY, cX = get_most_inner_point_from_a_binary_map(error_mask)
                
                # There might be more than one single point have furthest distance to background, randomly chose one
                random_idx = np.random.randint(0, len(cX))
                cX, cY = int(cX[random_idx]), int(cY[random_idx])
                # Get the current point into the prompt_list
                input_point_list.append((cX, cY))
                # print('cX, cY = {}, {}'.format(cX, cY))
                # print('shape of indicator mask is {}'.format(np.shape(indicator_mask)))

                # The value of such pixel is directly the label
                input_label_list.append(int(indicator_mask[cY, cX]))
                masks, scores, logits = prompt_with_multiple_points(predictor, 
                                            input_points=input_point_list, 
                                            input_labels=input_label_list,
                                            save_mask_path=save_mask_path, 
                                            save_mask_prefix=save_mask_prefix,
                                            logit_refine=last_logit,
                                            )
                
                # Choose according to the selection criteria
                index_chosen = choose_mask(masks, scores, 
                                            ground_truth=indicator_mask,
                                            oracle=choose_oracle)
                
                # Record the last prediciton mask as well as the logit
                last_prediction_mask = masks[index_chosen, :, :]
                last_logit = logits[index_chosen, :, :]
                last_logit = last_logit[None, :, :]
            
            # Save the input point and label list
            cur_img_prompt_point[i]['points'] = input_point_list
            cur_img_prompt_point[i]['labels'] = input_label_list
        
        # Save the prompt dictionary into main dict
        prompt_point_dict[img_name] = cur_img_prompt_point

        # This is for testing purpose that ends the process early
        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break

    # Save the whole dictionary down
    with open(os.path.join(save_mask_path, 
                        '{}_{}_choose_oracle_{}_iterative_prompt.pickle'.format(dataset, 
                                                                                num_point_prompt, 
                                                                                choose_oracle)), 
                           'wb') as handle:
                pickle.dump(prompt_point_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prompt_folder_with_bbox(mask_folder, bbox_df_file='bbox.csv',
                            img_folder = 'solar-pv',
                            max_img=999999):
    # Make the saving folder
    save_mask_path = 'solar_pv_bbox_prompt_save_{}'.format(mask_folder)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    # Load the points to be prompted
    print('...loading pickel of prompt points')
    df = pd.read_csv(os.path.join(mask_folder, bbox_df_file),
                    index_col=0)
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()
    prev_img_path = None

    # Loop over all the keys inside the prompt_point_dict
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Get image path
        img_path = os.path.join(img_folder, row['img_name'])
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue

        if img_path != prev_img_path: # Only if the image is different from last one
            # Load the image and transform color
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Why do we need to cvt color?
            # Set the predictor
            predictor.set_image(image)

        # Get the input points (they are all in the form of a list)
        save_mask_prefix = row['img_name'].split('.')[0] + '_prompt_ind_{}_size_{}'.format(row['prop_ind'], row['area'])
        input_bbox = process_str_bbox_into_bbox(row['bbox'])    # Adding permutation to make this aligned
        prompt_with_bbox(predictor, input_bbox, save_mask_path, save_mask_prefix,)
        
        # update the prev_img_path
        prev_img_path = img_path

        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break

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


def prompt_folder_with_point(mode, max_img=999999):
    # Make the saving folder
    save_mask_path = 'solar_pv_{}_prompt_save'.format(mode)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    # Load the points to be prompted
    print('...loading pickel of prompt points')
    with open('solar_pv_{}_prompt_dict.pickle'.format(mode), 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()

    # Loop over all the keys inside the prompt_point_dict
    for img_name in tqdm(prompt_point_dict.keys()):
        # Get image path
        img_path = os.path.join('solar-pv', img_name)
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue
        # Load the image and transform color
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Why do we need to cvt color?
        # Set the predictor
        predictor.set_image(image)

        # Get the input points (they are all in the form of a list)
        for ind, input_point in enumerate(prompt_point_dict[img_name]):
            save_mask_prefix = img_name.split('.')[0] + '_prompt_ind_{}_'.format(ind)
            input_point_np = np.reshape(np.array(input_point), [1, 2])
            prompt_with_point(predictor, input_point_np, save_mask_path, save_mask_prefix)
        
        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break

def prompt_folder_with_mask(prompt_mask_dir, mode='mask', 
                            mask_prop_dict={'choice':'step', 'mag':50}, 
                            mask_file_postfix='.tif', 
                            save_prompt=False, max_img=999999):
    """
    The function that prompts for the folder with all the masks inside
    """ 
    # Make the saving folder
    save_mask_path = 'solar_pv_{}_{}_{}_prompt_save'.format(mode, mask_prop_dict['choice'], mask_prop_dict['mag'])
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()

    # Loop over all the keys inside the prompt_point_dict
    for img_name in tqdm(os.listdir(prompt_mask_dir)):
        # Get image path
        img_path = os.path.join('solar-pv', img_name)
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue
        # Load the image and transform color
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Set the predictor
        predictor.set_image(image)
        
        # Process the mask (get connected component)
        mask = cv2.imread(os.path.join(prompt_mask_dir, img_name))     # Get the single dimension mask
        output = cv2.connectedComponentsWithStats(mask[:, :, 0], 4)
        # Get them into structured outputs
        (numLabels, labels, stats, centroids) = output
        if np.max(mask) > 1:    # This is to make the mask binary
            mask_binary = mask[:, :, 0] > 122
        else:
            mask_binary = mask[:, :, 0]
        mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1

        # Get the input points (they are all in the form of a list)
        for i in range(numLabels):
            prompt_mask_input = mask_mul_labels == (i+1)
            num_pixel = np.sum(prompt_mask_input)
            if num_pixel == 0: # Identify background and skip this
                continue
            # Get the save name sorted out
            save_mask_prefix = img_name.split('.')[0] + '_prompt_ind_{}_numPixel_{}'.format(i, num_pixel)
            
            # Make the prompt mask
            prompt_mask_input = prompt_mask_input.astype('float')
            prompt_mask = make_prompt_mask(prompt_mask_input, mask_prop_dict)
            if save_prompt:
                np.save(os.path.join(save_mask_path, save_mask_prefix+'_prompt_mask.npy'),prompt_mask)
            prompt_mask = cv2.resize(prompt_mask, (256, 256))
            prompt_mask = np.expand_dims(prompt_mask, 0)
            # Actual prompting the SAM
            prompt_with_mask(predictor, prompt_mask, save_mask_path, save_mask_prefix)
        
        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break

def make_prompt_mask(prompt_mask_input, mask_prop_dict):
    """
    The function to make the prompt mask from a given binary map into the logit space of SAM
    """
    mag = mask_prop_dict['mag']
    background = mask_prop_dict['background']
    if mask_prop_dict['choice'] == 'step':
        # A series of transformations to make it [-mag, mag] with shape [1, 256, 256]
        prompt_mask = prompt_mask_input*mag
    elif mask_prop_dict['choice'] == 'kernel':
        prompt_mask = prompt_mask_input*mag
        # kernel = np.array([[0, 1, 2, 1, 0],
        #                     [1, 2, 3, 2, 1],
        #                     [2, 3, 4, 3, 2],
        #                     [1, 2, 3, 2, 1],
        #                     [0, 1, 2, 1, 0],
        #                     ], dtype='float') # Gx + j*Gy
        kernel = np.ones((mask_prop_dict['kernel_size'],mask_prop_dict['kernel_size']))
        # kernel /= np.sum(kernel)/4
        prompt_mask = signal.convolve2d(prompt_mask_input, kernel, boundary='symm', mode='same')
    
    prompt_mask[prompt_mask == 0] = background
    prompt_mask = cv2.resize(prompt_mask, (256, 256))    
    return prompt_mask
    
        
if __name__ == '__main__':
    ###############################################
    # Prompting the center/random points for the solar PV#
    ###############################################
    # mode = 'center'
    # mode = 'random'

    # Single point experiment
    # prompt_folder_with_point(mode)

    # Finetuned experiment from output mask of previous model
    # prompt_folder_with_mask('solar_finetune_mask')
    # for mask_mg in [0.1, 0.5, 1, 5, 10, 20, 100, 200]:
    #     prompt_folder_with_mask('solar_finetune_mask', mask_magnitude=mask_mg)

    # prompt_folder_with_mask('solar_finetune_mask', mask_magnitude=200)
    # prompt_folder_with_mask('solar_finetune_mask', mask_magnitude=55, save_prompt=True, max_img=30)

    # Solar
    # prompt_folder_with_multiple_points(mode='iterative_10_points',
    #                                    num_point_prompt=10,
    #                                    max_img=999999, 
    #                                    dataset='solar_pv',
    #                                    mask_folder='datasets/solar_masks',
    #                                    img_folder='datasets/solar-pv',
    #                                    img_postfix='tif',
    #                                    mask_postfix='tif',
    #                                    size_limit=0,
    #                                    SAM_refine_feedback=True,
    #                                    choose_oracle=True)
    
    # inria_DG
    # prompt_folder_with_multiple_points(mode='iterative_10_points',
    #                                     num_point_prompt=10,
    #                                     max_img=999999, 
    #                                     dataset='inria_dg',
    #                                     mask_folder='datasets/Combined_Inria_DeepGlobe_650/patches',
    #                                     img_folder='datasets/Combined_Inria_DeepGlobe_650/patches',
    #                                     img_postfix='jpg',
    #                                     mask_postfix='png',
    #                                     size_limit=0,
    #                                     SAM_refine_feedback=True,
    #                                     choose_oracle=False)

    # DG road
    prompt_folder_with_multiple_points(mode='iterative_10_points',
                                        num_point_prompt=10,
                                        max_img=999999, 
                                        dataset='dg_road',
                                        mask_folder='datasets/DG_road/train',
                                        img_folder='datasets/DG_road/train',
                                        img_postfix='sat.jpg',
                                        mask_postfix='mask.png',
                                        size_limit=0,
                                        SAM_refine_feedback=True,
                                        choose_oracle=True)

    # Cloud
    # prompt_folder_with_multiple_points(mode='iterative_10_points',
    #                                     num_point_prompt=10,
    #                                     max_img=999999, 
    #                                     dataset='cloud',
    #                                     mask_folder='datasets/cloud/train_processed',
    #                                     img_folder='datasets/cloud/train_processed',
    #                                     img_postfix='img_',
    #                                     mask_postfix='gt_',
    #                                     size_limit=50,
    #                                     SAM_refine_feedback=True,
    #                                     choose_oracle=False)
    
    # mask_folder = 'solar_masks'
    # # mask_folder = 'solar_finetune_mask'
    # prompt_folder_with_bbox(mask_folder)

    
