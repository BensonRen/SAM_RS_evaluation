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

# Some global variables
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

def load_predictor():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

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

def prompt_with_multiple_points(predictor, input_points, save_mask_path, 
                                save_mask_prefix, png_mode=True): # Pickle mode is not saving space
    """
    Prompt the predictor with img and point pair and save the mask in the save_mask_name
    :param predictor: The SAM loaded predictor, which has already been set_image before
    :param img_path: The location of the image
    :param input_point: A single point of prompting
    :param save_mask_prefix: The prefix to save the mask
    :param save_mask_path: The position to save the mask
    :param input_label: This is for the signifying whether it is foreground or background
    """
    input_labels = np.ones(len(input_points))
    # Make inference using SAM
    masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,)
    
    if png_mode:
        save_name = os.path.join(save_mask_path, 
                                save_mask_prefix + '_{}_{}_{}.png'.format(scores[0],scores[1],scores[2]))
        cv2.imwrite(save_name, np.swapaxes(masks, 0, 2)*255)
        return 
    else:
        # make the save path
        save_name = os.path.join(save_mask_path, 
                                save_mask_prefix + '_{}_{}_{}.npy'.format(scores[0],scores[1],scores[2]))
        np.save(save_name, masks)

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

def prompt_folder_with_multiple_points(mode, num_point_prompt, max_img=999999):
    save_mask_path = 'solar_pv_{}_prompt_save_numpoint_{}'.format(mode, num_point_prompt)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    # Load the points to be prompted
    print('...loading pickel of prompt points')
    with open('point_prompt_pickles/solar_pv_{}_prompt.pickle'.format(mode), 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()
    print(predictor)

    # Loop over all the keys inside the prompt_point_dict
    for img_name in tqdm(prompt_point_dict.keys()):
        # Get image path
        img_path = os.path.join('datasets/solar-pv', img_name)
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
            # Take the random num_point_prompt points (they are random so just the first)
            input_point_np = input_point[:num_point_prompt, :]
            prompt_with_multiple_points(predictor, input_point_np, save_mask_path, save_mask_prefix)
        
        if max_img is not None:
            max_img -= 1
            if max_img < 0:
                break
    

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



    # Multiple points
    # for num_point_prompt in [5, 10, 20]:
    # for num_point_prompt in [30, 40, 50]:
    for num_point_prompt in [2]:
        prompt_folder_with_multiple_points(mode = 'multi_point_rand_50', 
                                        num_point_prompt=num_point_prompt)
        

    # mask_folder = 'solar_masks'
    # # mask_folder = 'solar_finetune_mask'
    # prompt_folder_with_bbox(mask_folder)

    
