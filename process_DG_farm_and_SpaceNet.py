import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import shutil

# For SpaceNet shapely operations and rastering the shapefile
from shapely.geometry.polygon import Polygon
import shapely.wkt
import rasterio.features

def get_class_label_dict(img_len=2448):
    """
    get from class label
    """
    label_class = pd.read_csv('datasets/DG_land/class_dict.csv')
    label_dict = {}
    for i in range(len(label_class)):
        rgb = np.array(label_class.iloc[i, 1:].astype('int').values)
        # mask = np.ones([img_len, img_len, 3]) * rgb
        # label_dict[label_class.iloc[i, 0]] = mask
        label_dict[label_class.iloc[i, 0]] = rgb
    # print(label_dict)
    return label_dict

def get_new_mask_folder(label_dict,
                        files = None,
                        src_folder='/home/sr365/SAM/datasets/DG_land/train',
                        tgt_folder='/home/sr365/SAM/datasets/DG_land/diff_train_masks',
                        ):
    # Loop over the label classes
    for label in label_dict.keys():
        tgt_task_folder = os.path.join(tgt_folder, label)
        if not os.path.exists(tgt_task_folder):
            os.makedirs(tgt_task_folder)        # Make the directory
    files = os.listdir(src_folder) if files is None else files
    # Always loop over the source folder
    for file in tqdm(files):
        if 'mask.png' not in file: # Only go through the masks
            continue
        mask = cv2.imread(os.path.join(src_folder, file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)        # Make it into RGB
        # pixel_count = 0
        for label in label_dict.keys():
            tgt_task_mask = os.path.join(tgt_folder, label, file)
            new_mask = mask == label_dict[label]
            new_mask = np.all(new_mask, axis=2) * 255
            # percent_pix = np.mean(new_mask)*100
            # pixel_count += percent_pix
            # print('this {} occupy {}\% of area'.format(label, 
            #                                            np.mean(new_mask)*100))
            cv2.imwrite(tgt_task_mask, new_mask)
        # print(pixel_count)
        # quit()

def parallel_get_new_mask_DG_land():
    class_dict = get_class_label_dict()
    folder = '/home/sr365/SAM/datasets/DG_land/train'
    src_folder='/home/sr365/SAM/datasets/DG_land/train',
    tgt_folder='/home/sr365/SAM/datasets/DG_land/diff_train_masks'
    all_files = [file for file in os.listdir(folder) if '.png' in file] # .png is for crop
    print(len(all_files))
    num_cpu = 50
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            args_list.append((class_dict, all_files[i::num_cpu]))
        # print((args_list))
        # print(len(args_list))
        output_dfs = pool.starmap(get_new_mask_folder, args_list)
    finally:
        pool.close()
        pool.join()

def parallelize_spacenet():
    shapefile = pd.read_csv('datasets/SpaceNet6/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    index_full = np.arange(len(shapefile))
    num_cpu = 50
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in range(num_cpu):
            input_indexs = list(index_full[i::num_cpu])
            args_list.append((input_indexs,))
        output_dfs = pool.starmap(Process_SpaceNet, args_list)
    finally:
        pool.close()
        pool.join()

def Process_SpaceNet(list_of_index, 
                     img_folder='datasets/SpaceNet6/PS-RGB',
                     save_folder='datasets/SpaceNet6/masks'):
    """
    Get the number of objects for current image
    """
    shapefile = pd.read_csv('datasets/SpaceNet6/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    sub_file = shapefile.iloc[list_of_index, :]
    for i in tqdm(range(len(sub_file))):
        # Get the image id and subsequently image name
        img_id = sub_file['ImageId'].values[i]
        if 'EMPTY' in sub_file['PolygonWKT_Pix'].values[i]:
            continue
        img_name = 'SN6_Train_AOI_11_Rotterdam_PS-RGB_{}.tif'.format(img_id)
        # Read image and get shape
        img = cv2.imread(os.path.join(img_folder, img_name))
        img_size = np.shape(img)[:2]
        plg = shapely.wkt.loads(sub_file['PolygonWKT_Pix'].values[i])
        x, y = plg.exterior.xy
        mask = rasterio.features.rasterize([plg], out_shape=img_size) * 255
        name = os.path.join(save_folder, 
                            img_name.replace('.', '_ObjId_{}.'.format(sub_file['TileBuildingId'].values[i])))
        cv2.imwrite(name, cv2.merge((mask,mask,mask)))    # Save the three channeel image

def combine_SpaceNet_per_object_mask_to_per_image_mask(mask_folder='datasets/SpaceNet6/masks',
                                                       save_folder='datasets/SpaceNet6/mask_per_img'):
    """
    Previously rasterized masks are per object and now combining them into a per image mask
    """
    if not os.path.isdir(save_folder):
        print('creating folder {}...'.format(save_folder))
        os.makedirs(save_folder)
    # First get a dictionary of img_name:->list(files)
    file_list = os.listdir(mask_folder)
    img_dict = {}
    for file in tqdm(file_list):
        img_name = file.split('_ObjId_')[0]
        if img_name not in img_dict:
            img_dict[img_name] = []
        img_dict[img_name].append(file)
    # combine those imgs
    for img_name in tqdm(img_dict.keys()):
        full_img = None
        for cur_mask in img_dict[img_name]:
            # print(cur_mask)
            img = cv2.imread(os.path.join(mask_folder, cur_mask))
            if full_img is None:
                full_img = img
            else:
                full_img += img
        # Do a union operation
        full_img = full_img > 0
        full_img = full_img.astype('int')
        full_img *= 255
        # save combined mask
        save_name = os.path.join(save_folder, img_name + '.tif')
        # print('saving to ', save_name)
        cv2.imwrite(save_name, full_img)

if __name__ == '__main__':
    combine_SpaceNet_per_object_mask_to_per_image_mask()
    # parallel_get_new_mask_DG_land()
    # parallelize_spacenet()
    