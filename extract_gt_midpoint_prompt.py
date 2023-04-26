import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle, random, torch, cv2, os
from multiprocessing import Pool
from collections import ChainMap

def k_th_occurence(k, list_of_binary):
    """
    Find the kth occurence of 1 of a binary list
    """
    # Alternative implementation (Take 3:32 for 8 instances)
    # return [i for i, n in enumerate(list_of_binary) if n == 1][k]

    # Original implementation (Take 3:32 for 8 instances)
    occurence = 0
    for i in range(len(list_of_binary)):
        if list_of_binary[i] == 1:
            occurence += 1
        if occurence == k:
            return i
    return 1/0  # Stop here, this is error if algo comes to this point

def k_list_th_occurence(k_list, list_of_binary):
    # Original implementation (Take 3:32 for 8 instances)
    occurence = 0
    output_list = np.zeros(len(k_list))
    cur_found = 0
    for i in range(len(list_of_binary)):
        if list_of_binary[i] == 1:
            occurence += 1
            if occurence in k_list:
                output_list[cur_found] = i
                cur_found += 1
        else:
            continue
        if cur_found == len(k_list):
            return output_list
    print('k_list', k_list)
    return 1/0  # Stop here, this is error if algo comes to this point

def get_kth_for_all(num_to_find_list, k_list, conected_comp_label_array, error_code=-99999):
    """
    The O(l*n) version of the original naive O(l*n*numLabels) algorithm
    We only go through the whole conected_comp_label_array once
    """
    assert len(num_to_find_list) == len(k_list), 'your size of num to find list is different from k_list'
    count_dict = {}
    # Setup the count list
    for ind, num in enumerate(num_to_find_list):
        count_dict[num] = k_list[ind]
    # count_list = np.zeros(len(num_to_find_list))
    return_list = np.ones(len(num_to_find_list)) * error_code # initialize to error_code
    # Loop over the whole label array
    for i in tqdm(range(len(conected_comp_label_array))):
        # Get current pixel label class
        cur_label = conected_comp_label_array[i]
        if cur_label not in count_dict:
            continue
        # Increase the current label count
        count_list[cur_label] += 1
        # If this is equal to the k_list target list, record current index i to the return list
        if count_list[cur_label] == k_list[cur_label]:
            return_list[cur_label] = i
    assert np.sum(return_list == error_code) == 0, 'Not all number found its k_th num, check!'
    return return_list

def get_list_of_centers(folder, img_name, output_dict, num_count_dict=None, 
                        verbose=False, size_limit=0):
    if 'cloud' in folder:
        size_limit = 50
    # Read the image mask
    if verbose:
        print('...reading image')
    mask = cv2.imread(os.path.join(folder, img_name))
    if verbose:
        print('shape of the image')
        print('...getting connected component')
    # Find the connected component
    output = cv2.connectedComponentsWithStats(
	        mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output

    if numLabels == 1:
        return None

    if np.max(mask > 1):    # This is to make the mask binary
        mask_binary = mask[:,:,0] > 122
    else:
        mask_binary = mask[:, :, 0]
    
    # Set up the midpoint outside class list
    midpoint_outside_index_list, k_list = [], []

    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1
    # Loop over the number of labels and check whether this is a background or foregroup
    # for i in tqdm(range(numLabels)):
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # print(num_pixel)
        # First identify background
        if num_pixel <= size_limit:
            continue
        # First lets make sure that the centroid position is inside the current class
        if labels[int(centroids[i, 1]), int(centroids[i, 0])] == i: # This means this is within current panel
            if img_name not in output_dict:
                output_dict[img_name] = []
            output_dict[img_name].append((int(centroids[i, 0]), int(centroids[i, 1])))
            # num_count_dict['center'] += 1
        else: # We need to find the closest component within such component
            midpoint_outside_index_list.append(i+1)
            k_list.append(num_pixel // 2)

    # If every object center is inside, end the function
    if len(midpoint_outside_index_list) == 0:
        return
    
    print('entering the center ouside part, going through {} of them'.format(len(midpoint_outside_index_list)))
    # Need to deal with objects that have class that its center outside itself
    # Start from getting the coordinate sequences
    l, w = np.shape(labels) # Get the size of the current image
    l_index = np.broadcast_to(np.arange(l), (w, l)).T
    w_index = np.broadcast_to(np.arange(w), (l, w))
    l_index_flat = np.reshape(l_index, [-1, 1])
    w_index_flat = np.reshape(w_index, [-1, 1])

    # for center_out_ind, k in tqdm(zip(midpoint_outside_index_list, k_list), total=len(k_list)):
    for center_out_ind, k in zip(midpoint_outside_index_list, k_list):
        indicator_mask = (mask_mul_labels == center_out_ind)
        list_of_component = np.reshape(indicator_mask, [-1, 1])
        mid_point_index = k_th_occurence(k, list_of_component)
        if img_name not in output_dict:
            output_dict[img_name] = []
        output_dict[img_name].append((int(w_index_flat[mid_point_index]), int(l_index_flat[mid_point_index])))
        assert indicator_mask[l_index_flat[mid_point_index], w_index_flat[mid_point_index]] == 1, 'in closest setting, x,y are possibly flipped'
        # num_count_dict['midpoint'] += 1
        # num_count_dict['set_of_non_concave_panel_imgs'].add(img_name)
    
    return output_dict


def get_list_of_random_inside_points(folder, img_name, output_dict, 
                                     size_limit=0):
    if 'cloud' in folder:
        size_limit = 50
    # Read the image mask
    mask = cv2.imread(os.path.join(folder, img_name))
    # Find the connected component
    output = cv2.connectedComponentsWithStats(
	        mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output
    if numLabels == 1:
        return
    
    if np.max(mask > 1):    # This is to make the mask binary
        mask_binary = mask[:,:,0] > 122
    else:
        mask_binary = mask[:, :, 0]
    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1

    # get the coordinate arrays
    l, w = np.shape(labels) # Get the size of the current image
    l_index = np.broadcast_to(np.arange(l), (w, l)).T
    w_index = np.broadcast_to(np.arange(w), (l, w))
    l_index_flat = np.reshape(l_index, [-1, 1])
    w_index_flat = np.reshape(w_index, [-1, 1])

    
    # Loop over the number of labels and check whether this is a background or foregroup
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # print(num_pixel)
        # First identify background
        if num_pixel <= size_limit:
            continue
        # Directly choose an random point
        indicator_mask = mask_mul_labels == (i + 1)
        list_of_component = np.reshape(indicator_mask, [-1, 1])
        random_index = random.randint(1, num_pixel)
        mid_point_index = k_th_occurence(random_index, list_of_component)
        if img_name not in output_dict:
            output_dict[img_name] = []
        output_dict[img_name].append((int(w_index_flat[mid_point_index]), int(l_index_flat[mid_point_index])))
        assert indicator_mask[l_index_flat[mid_point_index], w_index_flat[mid_point_index]] == 1, 'in closest setting, x,y are possibly flipped'
    return output_dict

def get_list_of_multiple_random_inside_points(folder, img_name, output_dict, num_points,
                                     size_limit=0):
    if 'cloud' in folder:
        size_limit = 50
    size_limit = max(size_limit, num_points)
    # Read the image mask
    mask = cv2.imread(os.path.join(folder, img_name))
    # Find the connected component
    output = cv2.connectedComponentsWithStats(
	        mask[:, :, 0], 4)
    # Get them into structured outputs
    (numLabels, labels, stats, centroids) = output
    if numLabels == 1:
        return
    
    if np.max(mask > 1):    # This is to make the mask binary
        mask_binary = mask[:,:,0] > 122
    else:
        mask_binary = mask[:, :, 0]
    mask_mul_labels = mask_binary * (labels + 1) # Label + 1 is to make sure they all start from 1

    # get the coordinate arrays
    l, w = np.shape(labels) # Get the size of the current image
    l_index = np.broadcast_to(np.arange(l), (w, l)).T
    w_index = np.broadcast_to(np.arange(w), (l, w))
    l_index_flat = np.reshape(l_index, [-1, 1])
    w_index_flat = np.reshape(w_index, [-1, 1])

    
    # Loop over the number of labels and check whether this is a background or foregroup
    for i in range(numLabels):
        num_pixel = np.sum(mask_mul_labels == (i+1))
        # print(num_pixel)
        # First identify background
        if num_pixel <= size_limit:
            continue
        # Directly choose an random point
        indicator_mask = mask_mul_labels == (i + 1)
        list_of_component = np.reshape(indicator_mask, [-1, 1])
        random_index = np.random.permutation(num_pixel-1) + 1
        random_index = random_index[:num_points]
        # random_index = np.random.randint(1, num_pixel, size=num_points)
        mid_point_indexs = k_list_th_occurence(random_index, list_of_component).astype('int')
        if img_name not in output_dict:
            output_dict[img_name] = []
        # print(np.shape(mid_point_indexs))
        # print(np.shape(w_index_flat[mid_point_indexs]))

        point_stack = np.concatenate([w_index_flat[mid_point_indexs], l_index_flat[mid_point_indexs]],
                                     axis=1)
        output_dict[img_name].append(point_stack)
        # output_dict[img_name].append((int(w_index_flat[mid_point_index]), int(l_index_flat[mid_point_index])))
        # assert indicator_mask[l_index_flat[mid_point_index], w_index_flat[mid_point_index]] == 1, 'in closest setting, x,y are possibly flipped'
    return output_dict

def get_prompt_for_list_for_center(folder, file_list, output_dict):
    for file in file_list:
        get_list_of_centers(folder=folder,img_name=file, 
                        output_dict=output_dict)
    return output_dict

def get_prompt_for_list_for_random(folder, file_list, output_dict):
    for file in file_list:
        get_list_of_random_inside_points(folder=folder,img_name=file, 
                        output_dict=output_dict)
    return output_dict

def get_prompt_for_list_for_random_with_k_points(folder, file_list, output_dict, k):
    for file in file_list:
        get_list_of_multiple_random_inside_points(folder=folder,img_name=file, 
                        output_dict=output_dict, num_points=k)
    return output_dict

if __name__ == '__main__':
    num_count_dict = {'center': 0, 'midpoint':0, 'set_of_non_concave_panel_imgs':set()}
    output_dict = {}

    # dataset = 'inria'
    # dataset = 'inria_DG'
    # dataset = 'Solar'
    # dataset = 'cloud'

    # mode = 'center'
    # mode = 'random'
    # mode = 'multi_point_rand_5'

    # Run one test: Solar
    # get_list_of_centers(folder='solar_masks',img_name='11ska655800_20_10.tif', 
                        # output_dict=output_dict, num_count_dict=num_count_dict )

    # Run one test: Inria
    # get_list_of_centers(folder='inria/train/gt',img_name='austin10.tif', 
    #                     output_dict=output_dict, num_count_dict=num_count_dict )


    # # Run all
    folder = 'datasets/solar_masks'                                # Solar-pv
    # folder = 'datasets/Combined_Inria_DeepGlobe_650/patches'       # Inria-DG
    # folder = 'datasets/cloud/train_processed'                      # Cloud
    # folder = 'datasets/DG_road/train'                              # DG_road
    # for file in os.listdir(folder):
    # k, k_limit = 0, 100
    k, k_limit = 0, 9999999999

    ######################################
    # Use some parallel computing method #
    ######################################
    all_files = [file for file in os.listdir(folder) if '.tif' in file]     # For Solar
    # all_files = [file for file in os.listdir(folder) if '.png' in file] # .png is for inria_DG, Road
    # all_files = [file for file in os.listdir(folder) if 'gt' in file] # .png is for cloud

    # Testing purpose
    # j = 5
    # print(all_files[j])
    # get_list_of_multiple_random_inside_points(folder, all_files[j], output_dict, 30,
    #                                  size_limit=0)
    # for key in output_dict.keys():
    #     points = output_dict[key]
    #     print(len(points))
    #     for i in range(len(points)):
    #         print(np.shape(points[i]))
    # # print(output_dict)
    # quit()


    num_cpu = 50
    k = 50
    mode = 'multi_point_rand_{}'.format(k)
    try: 
        pool = Pool(num_cpu)
        args_list = []
        if 'multi_point_rand_' in mode:
            # k = int(mode.split('_')[-1])
            for i in range(num_cpu):
                args_list.append((folder, all_files[i::num_cpu], output_dict, k))
        else:
            for i in range(num_cpu):
                args_list.append((folder, all_files[i::num_cpu], output_dict))
        # print((args_list))
        # print(len(args_list))
        if mode == 'center':
            output_dict = pool.starmap(get_prompt_for_list_for_center, args_list)
        elif mode == 'random':
            output_dict = pool.starmap(get_prompt_for_list_for_random, args_list)
        elif 'multi_point_rand' in mode:
            output_dict = pool.starmap(get_prompt_for_list_for_random_with_k_points, args_list)
    finally:
        pool.close()
        pool.join()
    output_dict = dict(ChainMap(*output_dict))

    with open('solar_pv_{}_prompt.pickle'.format(mode), 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('inria_DG_{}_prompt.pickle'.format(mode), 'wb') as handle:
    #             pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('DG_road_{}_prompt.pickle'.format(mode), 'wb') as handle:
    #             pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('cloud_{}_prompt.pickle'.format(mode), 'wb') as handle:
    #             pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    quit()

    

    ############################################
    # Use original sequential computing method #
    ############################################
    for file in tqdm(os.listdir(folder)):
        # print(file)
        # We would save the result of each image for inria dataset
        # if dataset == 'inria':
        #     num_count_dict = {'center': 0, 'midpoint':0, 'set_of_non_concave_panel_imgs':set()}
        #     output_dict = {}
        if dataset == 'inria' and '.tif' not in file:
            print('{} is not a .tif file, ignore')
            continue
        if dataset == 'Solar' and '.tif' not in file:
            print('{} is not a .tif file, ignore')
            continue
        if dataset == 'inria_DG' and '.png' not in file:
            # print('{} is not a .png file, ignore')
            continue
        if mode == 'center':
            get_list_of_centers(folder=folder,img_name=file, 
                        output_dict=output_dict, num_count_dict=num_count_dict)
        elif mode == 'random':
            get_list_of_random_inside_points(folder=folder,img_name=file, 
                        output_dict=output_dict)
        # if dataset == 'inria':
        # # Save the dictionary results down
        #     with open('inria_prompt_dicts/prompts_{}.pickle'.format(file.split('.')[0]), 'wb') as handle:
        #         pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     with open('inria_prompt_dicts/output_record_{}.pickle'.format(file.split('.')[0]), 'wb') as handle:
        #         pickle.dump(num_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        k +=1 
        if k > k_limit:
            break

    with open('inria_DG_{}_prompt.pickle'.format(mode), 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if mode == 'center':    # Only available for center mode
        with open('inria_DG_{}_prompt_output_record.pickle'.format(mode), 'wb') as handle:
            pickle.dump(num_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
