# Copied from https://www.kaggle.com/code/voglinio/from-masks-to-bounding-boxes
from tqdm import tqdm
import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from prompt_solar import show_box


from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def get_BBs_from_single_mask_img(mask_file, mask_folder):
    mask = cv2.imread(os.path.join(mask_folder, mask_file))[:, :, 0]
    lbl = label(mask)
    props = regionprops(lbl)
    return props

def draw_on_img(img_path, props, save_name):
    img = cv2.imread(img_path)
    f = plt.figure()
    plt.imshow(img)
    for prop in props:
        print(prop)
        print(prop.bbox)
        new_bbox = np.array(prop.bbox)
        new_bbox = new_bbox[[1,0,3,2]]
        show_box(new_bbox, plt.gca())
        # cv2.rectangle(img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
    
    plt.imshow(img)
    plt.savefig(save_name)

def extract_full_folder(mask_folder, save_df_file, img_limit = 9999999):
    """
    Calls the get_BBs_from_single_mask for all masks within a folder and output a df file
    """
    save_df = pd.DataFrame(columns=['img_name','prop_ind','bbox','centroid','area'])
    
    for file in tqdm(os.listdir(mask_folder)):
        if '.tif' not in file and '.png' not in file:
            continue
        # Extract the 
        props = get_BBs_from_single_mask_img(file, mask_folder)
        for ind, prop in enumerate(props):
            save_df.loc[len(save_df)] = [file, ind, np.array(prop.bbox), 
                                         np.array(prop.centroid), prop.area]
        
        # To do small scale tests
        img_limit -= 1
        if img_limit < 0:
            break
    save_df.to_csv(save_df_file)

if __name__ == '__main__':
    # mask_folder = './'
    # mask_file = '11ska625740_31_05.tif'
    # img_path = '11ska625740_31_05_original_img.tif'

    # # Get the props of bounding box
    # props = get_BBs_from_single_mask_img(mask_file, mask_folder)
    # print(props)
    
    # # Draw them 
    # draw_on_img(img_path, props, save_name='investigation/mask_to_BB/test.png')

    # mask_folder = 'solar_masks' # The GT solar pv masks
    # mask_folder = 'solar_finetune_mask' # The detector output solar pv masks
    # mask_folder = 'Combined_Inria_DeepGlobe_650/patches' # The GT inria_DG masks
    mask_folder = 'DG_road/train'                       # The GT for Inria Road
    extract_full_folder(mask_folder=mask_folder, 
                        save_df_file=os.path.join(mask_folder, 'bbox.csv'))
