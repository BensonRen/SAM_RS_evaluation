import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
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
    mask_path = os.path.join(gt_folder, gt_file)
    gt_mask = cv2.imread(mask_path)
    img_size = np.shape(gt_mask)[:2]      # Get the shape of the image without the channel number, for spaceNet
    if np.max(gt_mask) > 1:    # This is to make the mask binary
        mask_binary = gt_mask[:,:,0] > 122
    else:
        mask_binary = gt_mask[:, :, 0]
    # prompt_list = prompt_point_dict[gt_file]
    if 'SpaceNet6' not in img_folder:    
        # Process the gt mask 
        output = cv2.connectedComponentsWithStats(
                gt_mask[:, :, 0], 4)
        # Get them into structured outputs
        (numLabels, labels, stats, centroids) = output
        
        mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1
    else:
        numLabels, indicator_mask_list  = get_numLabels_and_indicator_mask_list_for_SpaceNet(mask_path, img_size)

    output_max_conf_mask = np.zeros_like(mask_binary)    # This is the output mask
    output_max_IOU_mask = np.zeros_like(mask_binary)    # This is the output mask
    for i in range(numLabels):
        # There might be one image losing, to make sure the program still runs, add this continue
        if gt_file.split('.')[0] not in all_files or str(i) not in all_files[gt_file.split('.')[0]]:
            continue
        if str(point_num) not in all_files[gt_file.split('.')[0]][str(i)]:
            continue
        if 'SpaceNet6' not in img_folder:           # SpaceNet doesn't use mask but PolyGon instead
            num_pixel = np.sum(mask_mul_labels == (i+1))
            # First identify background, also clouds smaller than 50 pixels are exclueded as well
            if  num_pixel == 0 or ('cloud' in dataset and num_pixel <=50):
                continue
            cur_gt_mask = labels == i
        else:
            cur_gt_mask = mask_binary
        mask_file = all_files[gt_file.split('.')[0]][str(i)][str(point_num)]
        cur_mask = cv2.imread(os.path.join(prompt_mask_folder, mask_file))    # Load the SAM-prompted mask
        try:
            cur_mask = np.swapaxes(cur_mask, 0, 2) > 0  # swapeed axes during saving so swap back
        except:
            print('There is a problem with the mask, ', os.path.join(prompt_mask_folder, mask_file))
            continue
        if 'RITM' in prompt_mask_folder: # This is RITM case
            conf_list = [0,0,0]
            cur_mask = np.swapaxes(cur_mask, 1, 2)  # RITM output masks by doing repeat at 3rd dimension instead of swapping... now we swap back...
        else:   # SAM would read the confidence from the mask file name
            _, _, conf_list = read_info_from_prmopt_mask_file(mask_file)
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
                             pixel_IOU_mode=False,
                             point_num=0,
                             prompt_point_dict=None):
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
    # print(prompt_point_dict)
    if file_list is None:
        all_files = [file for file in os.listdir(gt_folder) if mask_postfix in file and file in prompt_point_dict] # .png is for inria_DG
    else:
        all_files = file_list
    for file in tqdm(all_files):
        get_IOU_from_gt_mask_point_prompt(file, dataset, prompt_point_dict, save_df, 
                                            prompt_mask_folder=prompt_mask_folder,
                                            mode=mode, 
                                            pixel_IOU_mode=pixel_IOU_mode,
                                            mask_postfix=mask_postfix,
                                            gt_folder=gt_folder,
                                            point_num=point_num,)
    if file_list is None: # Then this is not parallel mode
        save_df.to_csv('{}_{}_{}_wise_IOU.csv'.format( dataset, mode, 
                'pixel' if pixel_IOU_mode else 'object'))
    return save_df

def prepare_mask_folder(prompt_mask_folder, savename='files_dict.pickle', redo=False):
    """
    Prepare the prompt mask output folder as follows to accelerate the process
    Save the list of files into a dictionary:
    all_files.pickle with below strcuture:
    all_files = {img_name_1: {prompt_index_1: {point_num_1: FileName}, ...}, ...}
    :param redo: Redo it even if the target output file is present
    """
    if os.path.exists(os.path.join(prompt_mask_folder, savename)) and not redo:  # This is processed already
        return
    all_files = {}
    print('pre-processing the folder for list of file')
    for file in tqdm(os.listdir(prompt_mask_folder)):
        if '.csv' in file or '.pickle' in file:     # Skip the irrelevant files
            continue
        if '_prompt_ind_' not in file:
            print(file)
        img_name = file.split('_prompt_ind_')[0]
        if img_name not in all_files:
            all_files[img_name] = {}
        prompt_ind = file.split('_prompt_ind_')[1].split('_point_num_')[0]
        if prompt_ind not in all_files[img_name]:
            all_files[img_name][prompt_ind] = {}
        point_num = file.split('point_num_')[1].split('_')[0].split('.')[0]     # split _ for SAM output, . for RTIM
        all_files[img_name][prompt_ind][point_num] = file
    with open(os.path.join(prompt_mask_folder, savename ),'wb') as handle:
        pickle.dump(all_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 

def parallel_multiple_gt_mask(prompt_mask_folder, mode,  dataset, choose_oracle,
                            gt_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches',
                            mask_postfix='.png', 
                            pixel_IOU_mode=False,
                            point_num=0,
                            num_cpu = 50,
                            SAM_prompt=True):
    """
    Parallelism of the ground truth prompting
    """
    prepare_mask_folder(prompt_mask_folder, redo=True)
    prompt_point_dict = get_prompt_dict(mode=mode, dataset=dataset, 
                                        prompt_mask_folder=prompt_mask_folder)
    if len(prompt_point_dict) == 0:
        print('The prompt_point_dict is empty!!! now doing evaluation mode without \
              verifying the images with the prompt point dict...')
        all_files = [file for file in os.listdir(gt_folder) if mask_postfix in file]
    else:
        all_files = [file for file in os.listdir(gt_folder) if mask_postfix in file and file in prompt_point_dict]
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
                            pixel_IOU_mode,
                            point_num,
                            prompt_point_dict))
        output_dfs = pool.starmap(process_multiple_gt_mask, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    if SAM_prompt:
        combined_df.to_csv('{}_{}_point_{}_{}_wise_IOU_oracle_{}.csv'.format( dataset, mode, point_num,
                        'pixel' if pixel_IOU_mode else 'object', choose_oracle))
    else:   # This is RITM case
        combined_df.to_csv('RITM_{}_{}_point_{}_{}_wise_IOU_oracle.csv'.format(dataset, mode, point_num,
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
    SAM_prompt_result_folder = 'SAM_output/bbox/{}_bbox_prompt_save_{}'.format(dataset, mask_folder)
    if file_list is None:
        df = pd.read_csv(os.path.join(mask_folder, bbox_csv_name), index_col=0)     # Read the bbox csv
        file_list = list(set(df['img_name'].values))    # use set to remove duplicates
    for file in tqdm(file_list):
        reg_exp = os.path.join(SAM_prompt_result_folder, 
                                               '{}*'.format(file.replace(mask_postfix, '')))
        # print(reg_exp)
        cur_mask_list = glob.glob(reg_exp)
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
                                             gt_mask_folder=None,
                                             img_postfix='',
                                            mask_postfix='.png',
                                             num_cpu=5):
    """
    Calls the "get_pixel_IOU_from_bbox_prompt_from_bboxcsv" in parallel manner
    """
    df = pd.read_csv(os.path.join(mask_folder, bbox_csv_name), index_col=0)     # Read the bbox csv
    file_list = list(set(df['img_name'].values))    # use set to remove duplicates
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((mask_folder, dataset, file_list[i::num_cpu],gt_mask_folder, img_postfix, mask_postfix))
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
        if len(prompt_file_reg) == 1:       # Paralleled run has multiple .pickle files
            print('There is only a single .pickle, reading this one')
            prompt_file = prompt_file_reg[0]
        else:
            print('There are {} pickle files, probably resulting from a paralleled run,\
                   combining them together here ...'.format(len(prompt_file_reg)))
            full_dict = {}
            for file in prompt_file_reg:
                if 'files_dict' in file:    # There might be a files_dict, ignore that
                    continue    
                with open(file, 'rb') as handle:
                    cur_dict = pickle.load(handle)
                full_dict.update(cur_dict)
            return full_dict
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
    # mode='iterative_10_points_random'
    num_point_prompt=10
    size_limit=0
    num_cpu = 40
    choose_oracle=True

    # Solar
    # dataset='solar_pv'
    # mask_folder='datasets/solar_masks'
    # img_folder='datasets/solar-pv'
    # img_postfix='tif'
    # mask_postfix='tif'
    # choose_oracle=True

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

    # Cloud
    # dataset='cloud'
    # mask_folder='datasets/cloud/train_processed'
    # img_folder='datasets/cloud/train_processed'
    # img_postfix='img_'
    # mask_postfix='gt_'
    # size_limit=50

    # Crop
    dataset='crop'
    img_folder ='datasets/crop/imgs'
    mask_folder = 'datasets/crop/masks_filled'
    img_postfix='.jpeg'
    mask_postfix='.png'
    size_limit=0
    choose_oracle=True
    
    # DG_land use
    # for land_type in [  'water', 'urban_land']: #'agriculture_land',]:#
    #     dataset='dg_land_{}'.format(land_type)
    #     mask_folder='datasets/DG_land/diff_train_masks/{}'.format(land_type)
    #     img_folder='datasets/DG_land/train'
    #     img_postfix='sat.jpg'
    #     mask_postfix='mask.png'
    #     choose_oracle=True
    
    # SpaceNet (True instance segmentation)
    # dataset='SpaceNet'
    # mask_folder='datasets/SpaceNet6/mask_per_img' # There is no actual mask folder
    # trained_detector_output_mask_folder = 'detector_predictions/SpaceNet/masks'
    # trained_detector_gt_mask_folder = 'detector_predictions/SpaceNet/gt'
    # gt_bbox_folder = 'datasets/SpaceNet6/SummaryData'
    # img_folder='datasets/SpaceNet6/PS-RGB'
    # img_postfix='.tif'
    # mask_postfix='.tif'
    # size_limit=0
    # choose_oracle=True

    # Scaled inria_dg test
    # for scale in [2, 4, 8]:
    #     dataset='scaled_{}x_inria_dg'.format(scale)
    #     img_folder='datasets/inria_dg_scaled/{}x_upscaled/imgs'.format(scale)
    #     mask_folder='datasets/inria_dg_scaled/{}x_upscaled/gt'.format(scale)
    #     img_postfix='jpg'
    #     mask_postfix='png'
    #     choose_oracle=True
    #     num_point_prompt=5 # For scaled dataset it only goes to 5 points

    # DG road scaled
    # for scale in [2, 4, 8]:
    #     # scale = 8 # scale in [2, 4, 8]:
    #     dataset='scaled_{}x_dg_road'.format(scale)
    #     img_folder='datasets/dg_road_scaled/{}x_upscaled/imgs'.format(scale)
    #     mask_folder='datasets/dg_road_scaled/{}x_upscaled/gt'.format(scale)
    #     img_postfix='sat.jpg'
    #     mask_postfix='mask.png'
    #     choose_oracle=True
    #     num_point_prompt=5

    ##########################
    # Evaluation of the SAM #
    ##########################
    for mode in ['iterative_10_points',]:
    # for mode in ['iterative_10_points_random', 'iterative_10_points']:
        # for choose_oracle in [True, False]:
    
        prompt_mask_folder = 'SAM_output/oracle_iterative/{}_{}_prompt_save_numpoint_{}_oracle_{}'.format(dataset, 
        # prompt_mask_folder = 'SAM_output/{}_{}_prompt_save_numpoint_{}_oracle_{}'.format(dataset, 
                                mode, 
                                num_point_prompt,
                                choose_oracle)
        # for pixel_IOU_mode in [True, False]:
        for pixel_IOU_mode in [True]:
            for i in range(num_point_prompt):  # 5 point for scaled test
                parallel_multiple_gt_mask(prompt_mask_folder=prompt_mask_folder,
                                        mode=mode,
                                        dataset=dataset,
                                        choose_oracle=choose_oracle,
                                        gt_folder=mask_folder,
                                        mask_postfix=mask_postfix,
                                        pixel_IOU_mode=pixel_IOU_mode,
                                        point_num=i,
                                        num_cpu=num_cpu,
                                            SAM_prompt=True)
    
    ##########################
    # Evaluation of the RITM #
    ##########################
    # prompt_mask_folder = 'RITM_output/RITM_{}_{}_prompt_save_numpoint_{}_oracle'.format(dataset, 
    #                                                                     mode, 
    #                                                                     num_point_prompt)
    # for pixel_IOU_mode in [True]:
    #     for i in range(10):
    #         parallel_multiple_gt_mask(prompt_mask_folder=prompt_mask_folder,
    #                                 mode=mode,
    #                                 dataset=dataset,
    #                                 choose_oracle=None,
    #                                 gt_folder=mask_folder,
    #                                 mask_postfix=mask_postfix,
    #                                 pixel_IOU_mode=pixel_IOU_mode,
    #                                 point_num=i,
    #                                 num_cpu=num_cpu,
    #                                 SAM_prompt=False)

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
    # The detector prediction
    # parallel_pixel_IOU_calc_from_bbox_prompt(trained_detector_output_mask_folder,
    #                                          gt_mask_folder=trained_detector_gt_mask_folder,
    #                                          dataset=dataset,
    #                                         img_postfix=img_postfix,
    #                                         mask_postfix=mask_postfix,
    #                                          num_cpu=10)

    # The ground truth bbox
    # parallel_pixel_IOU_calc_from_bbox_prompt(gt_bbox_folder,
    #                                          gt_mask_folder=mask_folder,
    #                                          dataset=dataset,
    #                                         img_postfix=img_postfix,
    #                                         mask_postfix=mask_postfix,
    #                                          num_cpu=40)

