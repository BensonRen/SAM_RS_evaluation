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
from multiprocessing import Pool

from misc_utils import *

from shapely.geometry.polygon import Polygon
import shapely.wkt
import rasterio.features


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

def extract_full_folder(mask_folder, file_list=None, save_df_file=None, 
                        img_limit = 9999999, size_limit=0):
    """
    Calls the get_BBs_from_single_mask for all masks within a folder and output a df file
    """

    if 'cloud' in mask_folder or 'detector_predictions/dg_road' in mask_folder:       # Skip the tiny ones in cloud dataset
        size_limit = 50
    save_df = pd.DataFrame(columns=['img_name','prop_ind','bbox','centroid','area'])
    
    file_list = os.listdir(mask_folder) if file_list is None else file_list 
    for file in tqdm(file_list):
        if 'detector_predictions' not in mask_folder:
            if '.tif' not in file and '.png' not in file and 'gt_patch' not in file:
                continue
        else:
            if '.csv' in file:
                continue
        # Extract the 
        props = get_BBs_from_single_mask_img(file, mask_folder)
        for ind, prop in enumerate(props):
            if prop.area <= size_limit:
                # print('size = {} <= {}, skip'.format(prop.area, size_limit))
                continue
            save_df.loc[len(save_df)] = [file, ind, np.array(prop.bbox), 
                                         np.array(prop.centroid), prop.area]
        
        # To do small scale tests
        img_limit -= 1
        if img_limit < 0:
            break
    if save_df_file:
        save_df.to_csv(save_df_file)
    return save_df

def extract_bbox_for_SpaceNet(ind_list=None,save_df_file=None, bbox_permutation=[1,0,3,2]):
    """
    The function to extract bounding box for SpaceNet dataset that does not have a ground truth mask structure
    but a .csv file with Shapely PolyGon instead
    :param bbox_permutation: This is funny, actually by using shapely polygon this is plg.bounds 
        gives correct sequence of bbox parameters that matches the setting of SAM. However, as we 
        would flip in during the prompting (as other, sub-sequent functions outputs the flipped one),
        we need to flip it first to make it comparable with its sub-sequent API
    """
    save_df = pd.DataFrame(columns=['img_name','prop_ind','bbox','centroid','area'])
    shapefile = pd.read_csv('datasets/SpaceNet6/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    
    plg_list = shapely.wkt.loads(shapefile['PolygonWKT_Pix'])
    if ind_list is None:
        ind_list = range(len(shapefile))
    for i in tqdm(ind_list):
        # Skip the empty images
        if 'EMPTY' in shapefile['PolygonWKT_Pix'].values[i]:
            continue;
        file = 'SN6_Train_AOI_11_Rotterdam_PS-RGB_{}.tif'.format(shapefile['ImageId'].values[i])
        ind = shapefile['TileBuildingId'].values[i]
        plg = plg_list[i]
        bbox = np.array(plg.bounds)
        bbox = bbox[bbox_permutation]
        save_df.loc[len(save_df)] = [file, ind, bbox.astype('int'), 
                                         None, int(plg.area)]
    if save_df_file:
        save_df.to_csv(save_df_file)
    return save_df

def parallel_extract_bbox_spacenet(save_df_file, num_cpu=30):
    shapefile = pd.read_csv('datasets/SpaceNet6/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    all_index = range(len(shapefile))
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((
                            all_index[i::num_cpu], ))
        output_dfs = pool.starmap(extract_bbox_for_SpaceNet, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    combined_df.to_csv(save_df_file)


def parallel_extract_full_folder(mask_folder, save_df_file):
    file_list = os.listdir(mask_folder) 
    num_cpu = 10
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((mask_folder, file_list[i::num_cpu]))
        output_dfs = pool.starmap(extract_full_folder, args_list)
    finally:
        pool.close()
        pool.join()
    combined_df = pd.concat(output_dfs)
    combined_df.to_csv(save_df_file)
    
if __name__ == '__main__':
    # mask_folder = './'
    # mask_file = '11ska625740_31_05.tif'
    # img_path = '11ska625740_31_05_original_img.tif'

    # # Get the props of bounding box
    # props = get_BBs_from_single_mask_img(mask_file, mask_folder)
    # print(props)
    
    # # Draw them 
    # draw_on_img(img_path, props, save_name='investigation/mask_to_BB/test.png')

    # mask_folder = 'datasets/solar_masks' # The GT solar pv masks
    # mask_folder = 'detector_predictions/solar_finetune_mask' # The detector output solar pv masks
    # mask_folder = 'datasets/Combined_Inria_DeepGlobe_650/patches' # The GT inria_DG masks
    # mask_folder = 'detector_predictions/inria_dg/masks'               # The inria_DG detector output mask
    # mask_folder = 'datasets/DG_road/train'                       # The GT for Inria Road
    # mask_folder = 'detector_predictions/dg_road/masks'                    # The detecotr output for DG road
    # mask_folder = 'datasets/cloud/train_processed'                       # The GT for Cloud
    # mask_folder = 'detector_predictions/cloud/masks'                       # The detecotr output for Cloud
    # mask_folder = 'datasets/crop/masks_filled'                        # The GT of crop
    # mask_folder = 'detector_predictions/crop_delineation_filled'      # The detector output for crop

    # mask_folder = 'detector_predictions/SpaceNet/masks'      # The detector output for SpaceNet
    
    #####
    # The DG_land dataset
    #####
    # DG_land_type_list = ['urban_land', 'water' ,'agriculture_land' ]
    # for DG_land_type in DG_land_type_list:
    #     # DG_land_type = 'urban_land' # 'water' #  'agriculture_land' 
    #     dataset = 'DG_land_{}'.format(DG_land_type)
    #     mask_folder = 'datasets/DG_land/diff_train_masks/{}'.format(DG_land_type) 

    #     parallel_extract_full_folder(mask_folder=mask_folder, 
    #                         save_df_file=os.path.join(mask_folder, 'bbox.csv'))

    # Sequential version
    # extract_full_folder(mask_folder=mask_folder, 
    #                     save_df_file=os.path.join(mask_folder, 'bbox.csv'))
    
    # Parallel version
    # parallel_extract_full_folder(mask_folder=mask_folder, 
    #                     save_df_file=os.path.join(mask_folder, 'bbox.csv'))

    # For SpaceNet
    # extract_bbox_for_SpaceNet(save_df_file='datasets/SpaceNet6/SummaryData/bbox.csv')

    # parallell version
    parallel_extract_bbox_spacenet(save_df_file='datasets/SpaceNet6/SummaryData/bbox.csv')