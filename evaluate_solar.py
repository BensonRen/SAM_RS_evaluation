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

def get_pixel_IOU_from_gt_mask(gt_file, prompt_point_dict, save_df, mode,
    gt_folder='solar_masks'):
    """
    The function to get the pixel IOU related information for a single gt_file
    """
    prompt_mask_folder = 'solar_pv_{}_prompt_save'.format(mode)
    # Read the gt mask
    gt_mask = cv2.imread(os.path.join(gt_folder, gt_file))
    prompt_list = prompt_point_dict[gt_file]
    # Process the gt mask 
    output = cv2.connectedComponentsWithStats(
	        gt_mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output
    assert numLabels > 1, 'your image is completely background but it is in the prompt_point_dict?'
    if np.max(gt_mask) > 1:    # This is to make the mask binary
        mask_binary = gt_mask[:,:,0] > 122
    else:
        mask_binary = gt_mask[:, :, 0]
    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1
    
    output_max_conf_mask = np.zeros_like(mask_binary)    # This is the output mask
    output_max_IOU_mask = np.zeros_like(mask_binary)    # This is the output mask
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # print(num_pixel)
        # First identify background
        if num_pixel == 0:
            continue
        matched = False
        for prompt_ind, prompt_point in enumerate(prompt_list):
            # Find the match of current prompt index
            if labels[prompt_point[1], prompt_point[0]] != i:
                continue
            matched = True
            mask_file_reg_exp = '{}_prompt_ind_{}_*.npy'.format(gt_file.replace('.tif',''), prompt_ind)
            # print('reg_exp=', mask_file_reg_exp)
            mask_file_list = glob.glob(os.path.join(prompt_mask_folder, mask_file_reg_exp))
            assert len(mask_file_list) == 1, 'Your mask file list length is not equal to 1: \n {}'.format(mask_file_list)
            mask_file = mask_file_list[0]
            cur_mask = np.load(mask_file) > 0
            # print(np.shape(cur_mask))
            _, _, conf_list = read_info_from_prmopt_mask_file(mask_file)
            cur_gt_mask = labels == i
            IoU_list = get_IoU_for_3masks(cur_gt_mask, cur_mask)
            # Take the index of the largest confidence or IOU
            ind_max_IOU = np.argmax(IoU_list)
            ind_max_conf = np.argmax(np.array(conf_list))
            output_max_IOU_mask += cur_mask[ind_max_IOU, :, :]
            output_max_conf_mask += cur_mask[ind_max_conf, :, :]
            break
        assert matched, 'There is patch in your gt_mask that does not match to your prompt point, stop!'
    # Do a simple union of them (original they are summed up together)
    output_max_IOU_mask = output_max_IOU_mask > 0
    output_max_conf_mask = output_max_conf_mask > 0
    _, max_IOU_num_intersection, max_IOU_num_union = IoU_single_object_mask(mask_binary, output_max_IOU_mask)
    _, max_conf_num_intersection, max_conf_num_union = IoU_single_object_mask(mask_binary, output_max_conf_mask)
    save_df.loc[len(save_df)] = [gt_file, max_IOU_num_intersection, max_IOU_num_union,  
                    max_conf_num_intersection, max_conf_num_union]
    return

def process_multiple_gt_mask(mode='center', file_list=None,
                             pixel_IOU_mode=False):
    gt_folder='solar_masks'
    # Setup the save_df
    if pixel_IOU_mode:
        save_df = pd.DataFrame(columns=['img_name', 'max_IOU_num_pixel_intersection', 
                    'max_IOU_num_pixel_union',  'max_conf_num_pixel_intersection',
                      'max_conf_num_pixel_union'])
    else:
        save_df = pd.DataFrame(columns=['img_name','prompt_ind', 
                                    'SAM_conf_0','SAM_conf_1','SAM_conf_2',
                                    'IoU_0','IoU_1','IoU_2', 'cur_object_size'])
    
    prompt_point_dict = get_prompt_dict(mode=mode)
    if file_list is None:
        all_files = [file for file in os.listdir(gt_folder) if '.tif' in file and '_33' not in file] # .tif is for solar_pv
    else:
        all_files = file_list
    for file in tqdm(all_files):
        if pixel_IOU_mode:
            get_pixel_IOU_from_gt_mask(file, prompt_point_dict, save_df, mode=mode)
        else:
            process_single_gt_mask(file, prompt_point_dict, save_df, mode=mode)
    if file_list is None: # Then this is not parallel mode
        if pixel_IOU_mode:
            save_df.to_csv('solar_pv_{}_pixel_wise_IOU.csv'.format(mode))
        else:
            save_df.to_csv('solar_pv_{}_object_wise_IOU.csv'.format(mode))
    return save_df

def parallel_multiple_gt_mask(mode, pixel_IOU_mode=False):
    folder='solar_masks'
    prompt_point_dict = get_prompt_dict(mode=mode)
    all_files = [file for file in os.listdir(folder) if '.tif' in file and file in prompt_point_dict and '_33' not in file] # .png is for inria_DG
    num_cpu = 40
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((mode, all_files[i::num_cpu], pixel_IOU_mode))
        # print((args_list))
        # print(len(args_list))
        output_dfs = pool.starmap(process_multiple_gt_mask, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    if pixel_IOU_mode:
        combined_df.to_csv('solar_pv_{}_pixel_wise_IOU.csv'.format(mode))
    else:
        combined_df.to_csv('solar_pv_{}_object_wise_IOU.csv'.format(mode))

def process_single_gt_mask(gt_file, prompt_point_dict, save_df, mode,
            gt_folder='solar_masks'):
    solar_prompt_mask_folder = 'solar_pv_{}_prompt_save'.format(mode)
    # Read the gt mask
    gt_mask = cv2.imread(os.path.join(gt_folder, gt_file))
    prompt_list = prompt_point_dict[gt_file]
    # Process the gt mask 
    output = cv2.connectedComponentsWithStats(
	        gt_mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output
    mask_binary = gt_mask[:,:,0] > 122
    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # print(num_pixel)
        # First identify background
        if num_pixel == 0:
            continue
        matched = False
        for prompt_ind, prompt_point in enumerate(prompt_list):
            # Find the match of current prompt index
            if labels[prompt_point[1], prompt_point[0]] != i:
                continue
            matched = True
            mask_file_reg_exp = '{}_prompt_ind_{}_*.npy'.format(gt_file.replace('.tif',''), prompt_ind)
            # print('reg_exp=', mask_file_reg_exp)
            mask_file_list = glob.glob(os.path.join(solar_prompt_mask_folder, mask_file_reg_exp))
            assert len(mask_file_list) == 1, 'Your mask file list length is not equal to 1: \n {}'.format(mask_file_list)
            mask_file = mask_file_list[0]
            cur_mask = np.load(mask_file)
            _, _, conf_list = read_info_from_prmopt_mask_file(mask_file)
            cur_gt_mask = labels == i
            IoU_list = get_IoU_for_3masks(cur_gt_mask, cur_mask)
            save_df.loc[len(save_df)] = [gt_file, prompt_ind, *conf_list, *IoU_list, num_pixel]
            break
        assert matched, 'There is patch in your gt_mask that does not match to your prompt point, stop!'

# def process_multiple_gt_mask(mode='center', gt_folder='solar_masks', img_limit=99999999):
#     # Setup the save_df
#     save_df = pd.DataFrame(columns=['img_name','prompt_ind', 
#                                     'SAM_conf_0','SAM_conf_1','SAM_conf_2',
#                                     'IoU_0','IoU_1','IoU_2', 'cur_object_size'])

#     # Below is for potential implementation of the pixel-wise IoU
#     # save_df = pd.DataFrame(columns=['img_name','max_conf_union','max_conf_intersection',
#     #                                 'max_conf_union','max_conf_intersection', 'TotalObjectSize'])
    
#     prompt_point_dict = get_prompt_dict(mode=mode)
#     img_count = 0
#     for file in tqdm(os.listdir(gt_folder)):
#         if '.tif' not in file or '_33' in file:         # Skipping all files with _33 inside (cuz not infered)
#             continue
#         process_single_gt_mask(file, prompt_point_dict, save_df, mode=mode)
#         img_count += 1
#         if img_count > img_limit:
#             break
#     # print(save_df)
#     save_df.to_csv('solar_pv_{}_IOU.csv'.format(mode))

def overlay_all_masks(img_name, mask_file, conf_list, input_point, 
                    solar_prompt_mask_folder = 'solar_pv_center_prompt_save'):
    image = cv2.imread(os.path.join('solar-pv', img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = np.load(os.path.join(solar_prompt_mask_folder, mask_file))
    scores = conf_list
    input_label = np.array([1])

    # The plotting process
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def get_prompt_dict(mode):
    with open('solar_pv_{}_prompt_dict.pickle'.format(mode), 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    return prompt_point_dict

def read_info_from_prmopt_mask_file(prompt_mask_file):
    mask_name = os.path.basename(prompt_mask_file)
    img_name = mask_name.split('_prompt')[0] + '.tif'
    mask_ind = int(mask_name.split('_prompt_ind_')[1].split('_')[0])
    conf_list = [float(i) for i in mask_name.replace('.npy','').split('_')[-3:]]
    return img_name, mask_ind, conf_list

if __name__ == '__main__':
    # process_multiple_gt_mask(mode='center')
    # process_multiple_gt_mask(mode='random')

    # Pixel IOU run
    print('running this')
    parallel_multiple_gt_mask(mode='random', pixel_IOU_mode=True)
    parallel_multiple_gt_mask(mode='center', pixel_IOU_mode=True)