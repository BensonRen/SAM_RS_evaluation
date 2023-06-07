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
                                       choose_oracle=False,
                                       predictor_is_SAM=True,
                                       parallel_number=None,
                                       parallel_index=None,
                                       skip_done=True):
    """
    The master function for prompting

    Parallel setting note: This function itself is NOT parallelized, but the setting would cut the target imgs to run
                            into pieces so one can run a bunch of them outside this function to reach parallism
    :param parallel_number: The total number of parallel programs to run at the same time
                            If this is set to None, then running in a non-parallel way
    :param parallel_index: This is the i-th instance of the parallel run
    :param skip_done: Check using files_dict.pickle to make sure it has not been done before
    """
    # Assert the parallel setting
    if parallel_number is not None:
        assert parallel_index is not None, 'Your parallel setting is wrong, check parallel related parameter input'

    # Set up the saving place
    if predictor_is_SAM:
        save_mask_path = '{}_{}_prompt_save_numpoint_{}_oracle_{}'.format(dataset, 
                                                                        mode, 
                                                                        num_point_prompt,
                                                                        choose_oracle)
    else:
         save_mask_path = 'RITM_{}_{}_prompt_save_numpoint_{}_oracle'.format(dataset, 
                                                                        mode, 
                                                                        num_point_prompt)
    
     # Read in the files pickle to get the current finished file dictionary
    if skip_done:
        done_dict_file = os.path.join(save_mask_path, 'files_dict.pickle' )
        if not os.path.exists(done_dict_file):
            done_dict = None
        else:
            with open(done_dict_file,'rb') as handle:   # Read only once all the files dictionary
                done_dict = pickle.load(handle)

    # Set the random flag by the mode name
    random_flag = True if 'random' in mode.lower() else False

    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    mask_name_list = [file for file in os.listdir(mask_folder) if mask_postfix in file]

    if parallel_number is not None:
        mask_name_list = mask_name_list[parallel_index::parallel_number]
        print('PARALLEL MODE ON! doing {} of them in total this is {}'.format(parallel_number, parallel_index))
    print('reading files from {}, matching name {} has {}'.format(mask_folder, mask_postfix, len(mask_name_list)))
    
    # Load predictor
    predictor = load_predictor(predictor_is_SAM)

    # Pre-allocate a dictionary to save the prompt point information
    prompt_point_dict = {}

    # Loop over all the keys inside the prompt_point_dict
    for mask_name in tqdm(mask_name_list):
        if done_dict is not None: # If this image has appeared once, just ignore all of them
            if mask_name.split('.')[0] in done_dict:
                continue 
        cur_img_prompt_point = {}
        # Get image path
        img_path = os.path.join(img_folder, mask_name.replace(mask_postfix, img_postfix))
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue
        # Load the image and transform color
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_size = np.shape(image)[:2]      # Get the shape of the image without the channel number, for spaceNet
        # Set the predictor
        set_image_predictor(predictor, image, predictor_is_SAM)

        if 'SpaceNet6' not in img_folder:           # SpaceNet doesn't use mask but PolyGon instead
            # Get the masks and connected components
            mask = cv2.imread(os.path.join(mask_folder, mask_name))
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
        else:
            numLabels, indicator_mask_list  = get_numLabels_and_indicator_mask_list_for_SpaceNet(img_path, img_size)

        # Loop over the each "objects" in this current image
        for i in range(numLabels):
            # Pre-allocate the dictionary to save
            cur_img_prompt_point[i] = {}
            if 'SpaceNet6' not in img_folder:           # SpaceNet doesn't use mask but PolyGon instead
                indicator_mask = mask_mul_labels == (i + 1)
                num_pixel = np.sum(indicator_mask)
                # First identify background
                if num_pixel <= size_limit:
                    continue
            else:       # THis is for SpaceNet only
                if indicator_mask_list is None:  # Some of the images does not contain any buildings
                    continue
                else:
                    indicator_mask = indicator_mask_list[i]

            # Set up the point list and the label list
            input_point_list, input_label_list = [], []
            last_logit = None   # Initialize the last logit to be None

            for j in range(num_point_prompt):
                save_mask_prefix = mask_name.split('.')[0] + '_prompt_ind_{}_point_num_{}'.format(i, j)
                if j == 0:  # The first prompt is special, because it does not have 
                    cY, cX = get_most_inner_point_from_a_binary_map(indicator_mask, random_flag)
                else: # The iterative experience where each time get a point from 
                    error_mask = np.uint8(np.bitwise_xor(indicator_mask, 
                                                         last_prediction_mask))
                    cY, cX = get_most_inner_point_from_a_binary_map(error_mask, random_flag)
                
                # There might be more than one single point have furthest distance to background, randomly chose one
                if len(cX) == 0:    # Edge case, when SAM/RITM gets current object PERFECTLY!
                    cX, cY = 0, 0
                else:
                    random_idx = np.random.randint(0, len(cX))
                    cX, cY = int(cX[random_idx]), int(cY[random_idx])

                #########################
                # The SAM way of things #
                #########################
                if predictor_is_SAM:
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
                
                ##########################
                # The RITM way of things #
                ##########################
                else:
                    # RITM uses CLICK class for each of its clicks, with label, corrds and indx
                    input_point_list.append(Click(is_positive=int(indicator_mask[cY, cX]),
                                                   coords=(cY, cX), 
                                                   indx = j))
                    input_label_list.append(int(indicator_mask[cY, cX]))    # This is just for recoriding, not useful
                    masks = prompt_RITM_with_multiple_points(predictor, input_point_list, image, 
                                                    indicator_mask,
                                                    save_mask_path=save_mask_path, 
                                                    save_mask_prefix=save_mask_prefix,)
                    last_prediction_mask = masks[:, :, 0]
                    

            # Save the input point and label list
            cur_img_prompt_point[i]['points'] = input_point_list
            cur_img_prompt_point[i]['labels'] = input_label_list
        
        # Save the prompt dictionary into main dict
        prompt_point_dict[mask_name] = cur_img_prompt_point

        # This is for testing purpose that ends the process early
        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break

    # Save the whole dictionary down
    save_name = os.path.join(save_mask_path, 
                        '{}_{}_choose_oracle_{}_iterative_prompt.pickle'.format(dataset, 
                                                                                num_point_prompt, 
                                                                                choose_oracle))
    if parallel_number is not None:         # For parallel running, change the saving filename
        save_name = save_name.replace('.','_par_{}_ind_{}.'.format(parallel_number, parallel_index))
    with open(save_name, 'wb') as handle:
                pickle.dump(prompt_point_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prompt_folder_with_bbox(mask_folder, 
                            dataset='solar_pv',
                            bbox_df_file='bbox.csv',
                            img_folder = 'solar-pv',
                            parallel_number=None,
                            parallel_index=None,
                            max_img=999999):
    
    # Assert the parallel setting
    if parallel_number is not None:
        assert parallel_index is not None, 'Your parallel setting is wrong, check parallel related parameter input'

    # Make the saving folder
    save_mask_path = '{}_bbox_prompt_save_{}'.format(dataset, mask_folder)
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


    if parallel_number is not None:
        sub_df = df.iloc[parallel_index::parallel_number, :]
        print('PARALLEL MODE ON! doing {} of them in total this is {}'.format(parallel_number, parallel_index))
    else:
        sub_df = df

    # Loop over all the keys inside the prompt_point_dict
    for ind, row in tqdm(sub_df.iterrows(), total=sub_df.shape[0]):
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

    # mask_folder = 'solar_masks'
    # # mask_folder = 'solar_finetune_mask'
    # prompt_folder_with_bbox(mask_folder)



    ###############################################
    # Multi point iterative prompting for SAM #####
    ###############################################
    # Shared parameters
    mode='iterative_10_points'
    # mode='iterative_10_points_random'
    num_point_prompt=10
    max_img=999999
    size_limit=0
    SAM_refine_feedback=True

    # Solar
    # dataset='solar_pv'
    # mask_folder='datasets/solar_masks'
    # img_folder='datasets/solar-pv'
    # img_postfix='tif'
    # mask_postfix='tif'
    # choose_oracle=False

    # inria_DG
    # dataset='inria_dg'
    # mask_folder='datasets/Combined_Inria_DeepGlobe_650/patches'
    # img_folder='datasets/Combined_Inria_DeepGlobe_650/patches'
    # img_postfix='jpg'
    # mask_postfix='png'
    # choose_oracle=True

    # DG road
    # dataset='dg_road'
    # mask_folder='datasets/DG_road/train'
    # img_folder='datasets/DG_road/train'
    # img_postfix='sat.jpg'
    # mask_postfix='mask.png'
    # choose_oracle=False

    # Cloud
    # dataset='cloud'
    # mask_folder='datasets/cloud/train_processed'
    # img_folder='datasets/cloud/train_processed'
    # img_postfix='img_'
    # mask_postfix='gt_'
    # size_limit=50
    # choose_oracle=True

    # Crop
    dataset='crop'
    img_folder ='datasets/crop/imgs'
    mask_folder = 'datasets/crop/masks_filled'
    img_postfix='.jpeg'
    mask_postfix='.png'
    size_limit=0
    choose_oracle=True
    
    # DG_land use
    # for land_type in [ 'water', 'urban_land','agriculture_land',]:
    #     dataset='dg_land_{}'.format(land_type)
    #     mask_folder='datasets/DG_land/diff_train_masks/{}'.format(land_type)
    #     img_folder='datasets/DG_land/train'
    #     img_postfix='sat.jpg'
    #     mask_postfix='mask.png'
    #     choose_oracle=True

    # SpaceNet (True instance segmentation)
    # dataset='SpaceNet'
    # mask_folder='datasets/SpaceNet6/PS-RGB' # There is no actual mask folder
    # img_folder='datasets/SpaceNet6/PS-RGB'
    # trained_detector_cropped_img_folder = 'detector_predictions/SpaceNet/cropped_imgs'
    # trained_detector_output_mask_folder = 'detector_predictions/SpaceNet/masks'
    # gt_bbox_folder = 'datasets/SpaceNet6/SummaryData'
    # img_postfix='.tif'
    # mask_postfix='.tif'
    # size_limit=0
    # choose_oracle=False
    
    # Inria_dg_scaled 
    # scale = 8 # scale in [2, 4, 8]:
    # dataset='scaled_{}x_inria_dg'.format(scale)
    # img_folder='datasets/inria_dg_scaled/{}x_upscaled/imgs'.format(scale)
    # mask_folder='datasets/inria_dg_scaled/{}x_upscaled/gt'.format(scale)
    # img_postfix='jpg'
    # mask_postfix='png'
    # choose_oracle=True

    # DG road scaled
    # scale = 8 # scale in [2, 4, 8]:
    # dataset='scaled_{}x_dg_road'.format(scale)
    # img_folder='datasets/dg_road_scaled/{}x_upscaled/imgs'.format(scale)
    # mask_folder='datasets/dg_road_scaled/{}x_upscaled/gt'.format(scale)
    # img_postfix='sat.jpg'
    # mask_postfix='mask.png'
    # choose_oracle=True

    prompt_folder_with_multiple_points(mode=mode,
                                    num_point_prompt=num_point_prompt,
                                    max_img=max_img, 
                                    dataset=dataset,
                                    mask_folder=mask_folder,
                                    img_folder=img_folder,
                                    img_postfix=img_postfix,
                                    mask_postfix=mask_postfix,
                                    size_limit=size_limit,
                                    SAM_refine_feedback=SAM_refine_feedback,
                                    choose_oracle=choose_oracle,
                                    predictor_is_SAM=True,
                                    parallel_number=10,
                                    parallel_index=9)
    
    ###############################################
    # Multi point iterative prompting for RITM #####
    ###############################################
    # prompt_folder_with_multiple_points(mode=mode,
    #                                    num_point_prompt=num_point_prompt,
    #                                    max_img=max_img, 
    #                                    dataset=dataset,
    #                                    mask_folder=mask_folder,
    #                                    img_folder=img_folder,
    #                                    img_postfix=img_postfix,
    #                                    mask_postfix=mask_postfix,
    #                                    size_limit=size_limit,
    #                                    SAM_refine_feedback=SAM_refine_feedback,
    #                                    choose_oracle=choose_oracle,
    #                                    predictor_is_SAM=False)
    

    #############################################################################################
    # Multi point iterative prompting for RITM with parallelism beacuse it is super slow... #####
    #############################################################################################
    # prompt_folder_with_multiple_points(mode=mode,
    #                                 num_point_prompt=num_point_prompt,
    #                                 max_img=max_img, 
    #                                 dataset=dataset,
    #                                 mask_folder=mask_folder,
    #                                 img_folder=img_folder,
    #                                 img_postfix=img_postfix,
    #                                 mask_postfix=mask_postfix,
    #                                 size_limit=size_limit,
    #                                 SAM_refine_feedback=SAM_refine_feedback,
    #                                 choose_oracle=choose_oracle,
    #                                 predictor_is_SAM=False,
    #                                 parallel_number=10,
    #                                 parallel_index=9)

    ###############################################
    # BBox prompt #####
    ###############################################
    # detector output one
    # prompt_folder_with_bbox(mask_folder=trained_detector_output_mask_folder, 
    #                         dataset=dataset,
    #                         img_folder = trained_detector_cropped_img_folder)

    # GT bbox
    # prompt_folder_with_bbox(mask_folder=gt_bbox_folder, 
    #                         dataset=dataset,
    #                         img_folder=img_folder,
    #                         parallel_number=10,
    #                         parallel_index=9,)