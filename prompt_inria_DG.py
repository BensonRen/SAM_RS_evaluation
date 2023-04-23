import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from tqdm import tqdm
import pickle
import pandas as pd
sys.path.append("..")
import re
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
                    input_label = np.array([1]), png_mode=True): # Pickle mode is not saving space
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

def prompt_folder_with_point(mode, max_img=999999):
    save_mask_path = 'inria_DG_{}_prompt_save'.format(mode)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)

    # Load the points to be prompted
    print('...loading pickel of prompt points')
    with open('point_prompt_pickles/inria_DG_{}_prompt.pickle'.format(mode), 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    
    # Load predictor
    print('...loading predictor')
    predictor = load_predictor()

    # Loop over all the keys inside the prompt_point_dict
    for img_name in tqdm(prompt_point_dict.keys()):
        # Get image path
        img_path = os.path.join('datasets/Combined_Inria_DeepGlobe_650/patches', img_name.replace('.png','.jpg'))
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

def prompt_folder_with_bbox(mask_folder, bbox_df_file='bbox.csv',
                            img_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches',
                            max_img=999999):
    # Make the saving folder
    save_mask_path = 'inria_DG_bbox_prompt_save_{}'.format(mask_folder)
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
        img_path = os.path.join(img_folder, row['img_name'].replace('.png','.jpg'))  # Note that inria_DG has .jpg for img and .png for mask
        # Make sure this image exist
        if not os.path.exists(img_path):
            print('Warning!!! {} does not exist, bypassing now'.format(img_path))
            continue

        if img_path != prev_img_path: # Only if the image is different from last one
            # Load the image and transform color
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


if __name__ == '__main__':
    ###############################################
    # Prompting the center/random points for the Inria_DG#
    ###############################################
    # prompt_folder_with_point(mode='random')
    # prompt_folder_with_point(mode='center')

    # Prompting with bbox
    # mask_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches'         # The ground truth boxes
    mask_folder = 'detector_predictions/inria_dg/masks'           # The detecotr output boxes
    # prompt_folder_with_bbox(mask_folder, )
    prompt_folder_with_bbox(mask_folder, img_folder=mask_folder.replace('masks', 'cropped_imgs'))
    
