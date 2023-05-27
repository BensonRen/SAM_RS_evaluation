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
from misc_utils import *


def get_IOU_from_gt_mask_point_prompt(gt_file, dataset, 
                                            prompt_point_dict, save_df, 
                                            mode, pixel_IOU_mode=True,
                                            prompt_mask_folder=None,
                                            mask_postfix='.png',
                                            point_num = 0, 
                                            gt_folder='datasets/Combined_Inria_DeepGlobe_650/patches'):
    """
    The function to get the pixel IOU related information for a single gt_file
    :param point_num: The i-th iterative point of such object
    """
    with open(os.path.join(prompt_mask_folder, 'files_dict.pickle' ),'rb') as handle:   # Read only once all the files dictionary
        all_files = pickle.load(handle)
    if prompt_mask_folder is None:
        prompt_mask_folder = 'SAM_output/{}_{}_prompt_save'.format(dataset, mode)
    # Read the gt mask
    gt_mask = cv2.imread(os.path.join(gt_folder, gt_file))
    prompt_list = prompt_point_dict[gt_file]
    # Process the gt mask 
    output = cv2.connectedComponentsWithStats(
	        gt_mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output
    if np.max(gt_mask) > 1:    # This is to make the mask binary
        mask_binary = gt_mask[:,:,0] > 122
    else:
        mask_binary = gt_mask[:, :, 0]
    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1
    
    output_max_conf_mask = np.zeros_like(mask_binary)    # This is the output mask
    output_max_IOU_mask = np.zeros_like(mask_binary)    # This is the output mask
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # First identify background
        if  num_pixel == 0:
            continue
        mask_file = all_files[gt_file.replace(mask_postfix,'')][str(i)][str(point_num)]
        cur_mask = cv2.imread(os.path.join(prompt_mask_folder, mask_file))    # Load the SAM-prompted mask
        cur_mask = np.swapaxes(cur_mask, 0, 2) > 0  # swapeed axes during saving so swap back
        _, _, conf_list = read_info_from_prmopt_mask_file(mask_file)
        cur_gt_mask = labels == i
        IoU_list = get_IoU_for_3masks(cur_gt_mask, cur_mask)
        if not pixel_IOU_mode:
            save_df.loc[len(save_df)] = [gt_file, i, *conf_list, *IoU_list, num_pixel]
            continue
        else:
            # Take the index of the largest confidence or IOU
            ind_max_IOU = np.argmax(IoU_list)
            ind_max_conf = np.argmax(np.array(conf_list))
            output_max_IOU_mask += cur_mask[ind_max_IOU, :, :, ]
            output_max_conf_mask += cur_mask[ind_max_conf, :, :]
    if not pixel_IOU_mode:  # Object IoU does not need the below calculations
        return
    # Do a simple union of them (original they are summed up together)
    output_max_IOU_mask = output_max_IOU_mask > 0
    output_max_conf_mask = output_max_conf_mask > 0
    _, max_IOU_num_intersection, max_IOU_num_union = IoU_single_object_mask(mask_binary, output_max_IOU_mask)
    _, max_conf_num_intersection, max_conf_num_union = IoU_single_object_mask(mask_binary, output_max_conf_mask)
    save_df.loc[len(save_df)] = [gt_file, max_IOU_num_intersection, max_IOU_num_union,  
                    max_conf_num_intersection, max_conf_num_union, np.sum(mask_binary > 0)]
    return
          
def process_multiple_gt_mask(prompt_mask_folder,
                            mode='center',
                            dataset='inria_DG',
                            file_list=None,
                            gt_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches',
                            mask_postfix='.png', 
                             pixel_IOU_mode=False):
    # Setup the save_df
    if pixel_IOU_mode:
        save_df = pd.DataFrame(columns=['img_name', 'max_IOU_num_pixel_intersection', 
                    'max_IOU_num_pixel_union',  'max_conf_num_pixel_intersection',
                      'max_conf_num_pixel_union','cur_object_size'])
    else:
        save_df = pd.DataFrame(columns=['img_name','prompt_ind', 
                                    'SAM_conf_0','SAM_conf_1','SAM_conf_2',
                                    'IoU_0','IoU_1','IoU_2', 'cur_object_size'])
    # Pre-process the prompt_mask_folder to accelerate the whole process
    prepare_mask_folder(prompt_mask_folder)
    prompt_point_dict = get_prompt_dict(mode=mode, dataset=dataset, 
                                        prompt_mask_folder=prompt_mask_folder)
    if file_list is None:
        all_files = [file for file in os.listdir(gt_folder) if mask_postfix in file and file in prompt_point_dict] # .png is for inria_DG
    else:
        all_files = file_list
    for file in tqdm(all_files):
        get_IOU_from_gt_mask_point_prompt(file, dataset, prompt_point_dict, save_df, 
                                          prompt_mask_folder=prompt_mask_folder,
                                            mode=mode, pixel_IOU_mode=pixel_IOU_mode)
    if file_list is None: # Then this is not parallel mode
        save_df.to_csv('{}_{}_{}_wise_IOU.csv'.format( dataset, mode, 
                'pixel' if pixel_IOU_mode else 'object'))
    return save_df

def prepare_mask_folder(prompt_mask_folder, savename='files_dict.pickle'):
    """
    Prepare the prompt mask output folder as follows to accelerate the process
    Save the list of files into a dictionary:
    all_files.pickle with below strcuture:
    all_files = {img_name_1: {prompt_index_1: {point_num_1: FileName}, ...}, ...}
    """
    if os.path.exists(os.path.join(prompt_mask_folder, savename)):  # This is processed already
        return
    all_files = {}
    print('pre-processing the folder for list of file')
    for file in tqdm(os.listdir(prompt_mask_folder)):
        if '.csv' in file or '.pickle' in file:     # Skip the irrelevant files
            continue
        img_name = file.split('_prompt_ind_')[0]
        if img_name not in all_files:
            all_files[img_name] = {}
        prompt_ind = file.split('_prompt_ind_')[1].split('_point_num_')[0]
        if prompt_ind not in all_files[img_name]:
            all_files[img_name][prompt_ind] = {}
        point_num = file.split('point_num_')[1].split('_')[0]
        all_files[img_name][prompt_ind][point_num] = file
    with open(os.path.join(prompt_mask_folder, savename ),'wb') as handle:
        pickle.dump(all_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 

def parallel_multiple_gt_mask(prompt_mask_folder, mode,  dataset, 
                            gt_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches',
                            mask_postfix='.png', 
                            pixel_IOU_mode=False):
    """
    Parallelism of the ground truth prompting
    """
    prepare_mask_folder(prompt_mask_folder)
    prompt_point_dict = get_prompt_dict(mode=mode, dataset=dataset, 
                                        prompt_mask_folder=prompt_mask_folder)
    all_files = [file for file in os.listdir(gt_folder) if mask_postfix in file and file in prompt_point_dict] # .png is for inria_DG
    num_cpu = 50
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((prompt_mask_folder, 
                            mode, 
                            dataset,
                            all_files[i::num_cpu], 
                            gt_folder,
                            mask_postfix, 
                            pixel_IOU_mode))
        output_dfs = pool.starmap(process_multiple_gt_mask, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    combined_df.to_csv('{}_{}_{}_wise_IOU.csv'.format( dataset, mode, 
                    'pixel' if pixel_IOU_mode else 'object'))

def get_pixel_IOU_from_bbox_prompt_from_bboxcsv(mask_folder, dataset,
                                                file_list=None, 
                                                gt_folder='datasets/Combined_Inria_DeepGlobe_650/patches',
                                                img_postfix='',
                                                mask_postfix='.png',
                                                bbox_csv_name='bbox.csv',
                                                img_size=(512, 512)):
    """
    The function that evaluates the pixel-wise IoU for each image in the file_list
    """
    save_df = pd.DataFrame(columns=['img_name','intersection','union', 'cur_object'])
    SAM_prompt_result_folder = 'SAM_output/{}_bbox_prompt_save_{}'.format(dataset, mask_folder.replace('datasets/',''))
    if file_list is None:
        df = pd.read_csv(os.path.join(mask_folder, bbox_csv_name), index_col=0)     # Read the bbox csv
        file_list = list(set(df['img_name'].values))    # use set to remove duplicates
    for file in tqdm(file_list):
        cur_mask_list = glob.glob(os.path.join(SAM_prompt_result_folder, 
                                               '{}*'.format(file.replace(mask_postfix, img_postfix))))
        gt_mask_path = os.path.join(gt_folder, file)
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path)[:, :, 0] > 0
            gt_mask = gt_mask.astype('uint8')
        else:
            print('This mask does not exist {}'.format(gt_mask_path))
            gt_mask = np.zeros(img_size)
        mask_union = np.zeros_like(gt_mask)
        for mask_file in cur_mask_list:
            mask = cv2.imread(mask_file)
            if len(np.shape(mask)) == 3:
                mask = mask[:, :, 0]
            mask_union += mask  # Sum to make union
        mask_union = mask_union > 0     # Threshold to make binary
        _, intersection, union = IoU_single_object_mask(gt_mask, mask_union)
        save_df.loc[len(save_df)] = [file, intersection, union, np.sum(gt_mask > 0)]
    return save_df

def parallel_pixel_IOU_calc_from_bbox_prompt(mask_folder,
                                            dataset,
                                             bbox_csv_name='bbox.csv',
                                             gt_mask_folder=None):
    """
    Calls the "get_pixel_IOU_from_bbox_prompt_from_bboxcsv" in parallel manner
    """
    df = pd.read_csv(os.path.join(mask_folder, bbox_csv_name), index_col=0)     # Read the bbox csv
    file_list = list(set(df['img_name'].values))    # use set to remove duplicates
    num_cpu = 50
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((mask_folder, file_list[i::num_cpu],gt_mask_folder))
        output_dfs = pool.starmap(get_pixel_IOU_from_bbox_prompt_from_bboxcsv, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    combined_df.to_csv('{}_{}_pixel_wise_IOU_{}.csv'.format(dataset, 'bbox_prompt', 
                                                                   mask_folder.replace('/','')))

def get_prompt_dict(mode, dataset, savepos='point_prompt_pickles', prompt_mask_folder=None):
    """
    Retrieve the prompt dictionary
    :param: savepos: Previously all the point prompt pickles are all in the default place,
    However new multi-point iterative process saves the point prompt pickle file under the SAM output folder
    """
    prompt_file = os.path.join(savepos, '{}_{}_prompt.pickle'.format(dataset, mode))
    if not os.path.exists(prompt_file) and prompt_mask_folder is not None:
        print('{} not found, trying in-folder savepos...'.format(prompt_file))
        prompt_file_reg = glob.glob(os.path.join(prompt_mask_folder, '{}*.pickle'.format(dataset)))
        assert len(prompt_file_reg) == 1, 'there is more than 1 .pickle file in the prompt_mask_folder with dataset prefix'
        prompt_file = prompt_file_reg[0]
    print('loading prompt point dict...')
    with open(prompt_file, 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    return prompt_point_dict

def read_info_from_prmopt_mask_file(prompt_mask_file):
    """
    The function that parse the filename of prompt mask into confidence list, mask id and image name
    """
    mask_name = os.path.basename(prompt_mask_file)
    img_name = mask_name.split('_prompt')[0] + '.tif'
    mask_ind = int(mask_name.split('_prompt_ind_')[1].split('_')[0])
    conf_list = [float(i) for i in mask_name.replace('.png','').split('_')[-3:]]
    return img_name, mask_ind, conf_list


if __name__ == '__main__':
    # Shared parameters
    mode='iterative_10_points'
    num_point_prompt=10
    size_limit=0

    # Solar
    # dataset='solar_pv'
    # mask_folder='datasets/solar_masks'
    # img_folder='datasets/solar-pv'
    # img_postfix='tif'
    # mask_postfix='tif'

    # inria_DG
    dataset='inria_dg'
    mask_folder='datasets/Combined_Inria_DeepGlobe_650/patches'
    img_folder='datasets/Combined_Inria_DeepGlobe_650/patches'
    img_postfix='jpg'
    mask_postfix='png'
    choose_oracle=False

    # DG road
    # dataset='dg_road'
    # mask_folder='datasets/DG_road/train'
    # img_folder='datasets/DG_road/train'
    # img_postfix='sat.jpg'
    # mask_postfix='mask.png'

    # Cloud
    # dataset='cloud'
    # mask_folder='datasets/cloud/train_processed'
    # img_folder='datasets/cloud/train_processed'
    # img_postfix='img_'
    # mask_postfix='gt_'
    # size_limit=50
    
    # DG_land use
    # for land_type in ['agriculture_land', 'water', 'urban_land']:
    #     dataset='dg_land_{}'.format(land_type)
    #     mask_folder='datasets/DG_land/diff_train_masks/{}'.format(land_type)
    #     img_folder='datasets/DG_land/train'
    #     img_postfix='sat.jpg'
    #     mask_postfix='mask.png'

    prompt_mask_folder = 'SAM_output/{}_{}_prompt_save_numpoint_{}_oracle_{}'.format(dataset, 
                                                                        mode, 
                                                                        num_point_prompt,
                                                                        choose_oracle)

    parallel_multiple_gt_mask(prompt_mask_folder=prompt_mask_folder,
                             gt_folder=mask_folder,
                             mode=mode,
                             dataset=dataset)
    
    # Serial processing
    # process_multiple_gt_mask(mode='center')
    # process_multiple_gt_mask(mode='random')

    # parallel processing
    # parallel_multiple_gt_mask(mode='center')
    # parallel_multiple_gt_mask(mode='random')

    # parallel processing of the pixel IOU value
    # parallel_multiple_gt_mask(mode='center', pixel_IOU_mode=True)
    # parallel_multiple_gt_mask(mode='random', pixel_IOU_mode=True)

    # parallel processing of the object IOU value
    # parallel_multiple_gt_mask(mode='center', pixel_IOU_mode=False)
    # parallel_multiple_gt_mask(mode='random', pixel_IOU_mode=False)


    # multiple points
    # for num_point_prompt in [5, 10, 20, 30, 40, 50]:
    # for num_point_prompt in [2, 3]:
    #     # parallel_multiple_gt_mask(mode='multi_point_{}'.format(num_point_prompt), pixel_IOU_mode=True)
    #     parallel_multiple_gt_mask(mode='multi_point_{}'.format(num_point_prompt), pixel_IOU_mode=False)


    # parallel processing of the pxiel IOU for BBox prompt
    # parallel_pixel_IOU_calc_from_bbox_prompt('datasets/Combined_Inria_DeepGlobe_650/patches',
    #                                          gt_mask_folder='datasets/Combined_Inria_DeepGlobe_650/patches')
    # parallel_pixel_IOU_calc_from_bbox_prompt('detector_predictions/inria_dg')

    # For the detector prediction
    # mask_folder = 'detector_predictions/inria_dg/masks'
    # parallel_pixel_IOU_calc_from_bbox_prompt(mask_folder,
    #                                          gt_mask_folder=mask_folder.replace('masks', 'gt'))

