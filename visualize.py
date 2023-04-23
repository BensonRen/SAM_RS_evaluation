import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
from tqdm import tqdm
from multiprocessing import Pool
import shutil

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_bw_from_1d_mask(mask, reference_img):
    G = np.zeros_like(reference_img)
    G[mask>0.5] = [255,255,255] 
    return G

def plot_scatter(max_IoU_list, pixel_list,):
    # f = plt.figure(figsize=[5,5])
    alpha = 0.01 if len(pixel_list) > 1e5 else 0.05
    # print(alpha)
    plt.scatter(max_IoU_list,pixel_list, alpha=alpha, s=5)
    plt.xlim([0, 1])
    plt.yscale('log')
    
def visualize_single_mask(SAM_pred_folder,
                          SAM_pred_mask_name,
                          gt_mask_folder,
                          img_folder,
                          save_prefix,
                          max_IoU_list,
                          pixel_list,
                          oracle_iou,
                          pixel_size,
                          savefolder='sample_imgs',
                          move_img=False,
                          ):
    """
    Visualize a 1x3 
    """
    img_prefix = SAM_pred_mask_name.split('_prompt')[0]
    # Get original img
    img_re = os.path.join(img_folder, 
                          img_prefix.replace('mask','').replace('gt_patch','img_patch')+'*')
    if 'DG' in SAM_pred_folder or 'dg' in SAM_pred_folder:
        img_re += '.jpg'
    img_name = glob.glob(img_re)[0]
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the current mask
    cur_mask_name = os.path.join(SAM_pred_folder, SAM_pred_mask_name)
    # print(cur_mask_name)
    cur_mask_name = glob.glob(cur_mask_name + '*')[0]
    if '.png' in cur_mask_name:
        cur_mask = cv2.imread(cur_mask_name)
        cur_mask = np.swapaxes(cur_mask, 0, 2) > 0
    elif '.npy' in cur_mask_name:
        cur_mask = np.load(cur_mask_name)

        # Get gt_mask
    gt_mask_re = os.path.join(gt_mask_folder, img_prefix)
    # print(gt_mask_re)
    gt_mask_name = glob.glob(gt_mask_re + '*')
    if 'DG' in SAM_pred_folder or 'dg' in SAM_pred_folder:
        img_re += '.png'
    if len(gt_mask_name) == 0:
        gt_mask = np.zeros_like(img)[:, :, 0]
    else:
        gt_mask_name = gt_mask_name[0]
    gt_mask = cv2.imread(gt_mask_name)
    if np.max(gt_mask) == 1:
        gt_mask *= 255
    if move_img:
        gt_dest = os.path.join(savefolder, os.path.basename(gt_mask_name))
        img_dest = os.path.join(savefolder, os.path.basename(img_name))
        if not os.path.exists(gt_dest):
            shutil.copyfile(gt_mask_name, gt_dest)
        if not os.path.exists(img_dest):
            shutil.copyfile(img_name, img_dest)
        for i in range(3):
            f = plt.figure()
            cur_mask_bw = get_bw_from_1d_mask(cur_mask[i, :, :], img)
            plt.imshow(cur_mask_bw)
            plt.axis('off')
            plt.savefig(os.path.join(savefolder, 
                                     os.path.basename(cur_mask_name).replace('.','_mask_{}.'.format(i))))
            plt.close('all')
        return
    # Plotting function
    f = plt.figure(figsize=[15,10])
    ax = plt.subplot(231)
    plt.imshow(img)
    plt.axis('off')
    ax = plt.subplot(232)
    plt.imshow(gt_mask)
    plt.axis('off')
    ax = plt.subplot(233)
    plot_scatter(max_IoU_list,pixel_list,)
    plt.scatter(oracle_iou, pixel_size, s=10, c='r')

    for i in range(3):
        ax = plt.subplot(int('23{}'.format(i+4)))
        cur_mask_bw = get_bw_from_1d_mask(cur_mask[i, :, :], img)
        plt.imshow(cur_mask_bw)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, 
                             save_prefix + SAM_pred_mask_name.split('.')[0] + '.png'),
                transparent=True, dpi=300)
    plt.close('all')

def visualize_solar(max_img=100):
    # First lets read the center prompt result.csv
    df = pd.read_csv('result_IOUs/solar_pv_center_object_wise_IOU.csv', 
                     index_col=0)
    
    scatter_arr = np.load('scatter_list/solar_pv_scatter_list.npy')
    max_IoU_list, pixel_list = scatter_arr[:, 0], scatter_arr[:, 1]

    for i in tqdm(range(min(len(df), max_img))):
        SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
                                                       df['prompt_ind'].values[i])
        oracle_iou = max_IoU_list[i]
        pixel_size = pixel_list[i]

        visualize_single_mask(SAM_pred_folder='SAM_output/solar_pv_center_prompt_save',
                            SAM_pred_mask_name=SAM_pred_mask_name,
                            gt_mask_folder='datasets/solar_masks',
                            img_folder='datasets/solar-pv',
                            save_prefix='test',
                            max_IoU_list=max_IoU_list, 
                            pixel_list=pixel_list,
                            oracle_iou=oracle_iou,
                            pixel_size=pixel_size,
                            savefolder='sample_imgs/solar_center'
                            )

def visualize_inria_dg(max_img=1000, num_cpu=50, move_img=False):
     # First lets read the center prompt result.csv
    df = pd.read_csv('result_IOUs/inria_DG_center_object_wise_IOU.csv', 
                     index_col=0)
    
    scatter_arr = np.load('scatter_list/inria_DG_scatter_list.npy')
    max_IoU_list, pixel_list = scatter_arr[:, 0], scatter_arr[:, 1]
    
    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in tqdm(range(min(len(df), max_img))):
            SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
                                                       df['prompt_ind'].values[i])
            oracle_iou = max_IoU_list[i]
            pixel_size = pixel_list[i]
            args_list.append(('SAM_output/inria_DG_center_prompt_save',
                            SAM_pred_mask_name,
                            'datasets/Combined_Inria_DeepGlobe_650/patches',
                            'datasets/Combined_Inria_DeepGlobe_650/patches',
                            'test',
                            max_IoU_list, 
                            pixel_list,
                            oracle_iou,
                            pixel_size,
                            'sample_imgs/inria_dg_center',
                            move_img))
        pool.starmap(visualize_single_mask, args_list)
    finally:
        pool.close()
        pool.join()

    # for i in tqdm(range(min(len(df), max_img))):
    #     if df['prompt_ind'].values[i] > 5:      # Only output 5 masks per image
    #         continue
    #     SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
    #                                                    df['prompt_ind'].values[i])
    #     oracle_iou = max_IoU_list[i]
    #     pixel_size = pixel_list[i]

    #     visualize_single_mask(SAM_pred_folder='SAM_output/inria_DG_center_prompt_save',
    #                         SAM_pred_mask_name=SAM_pred_mask_name,
    #                         gt_mask_folder='datasets/Combined_Inria_DeepGlobe_650/patches',
    #                         img_folder='datasets/Combined_Inria_DeepGlobe_650/patches',
    #                         save_prefix='test',
    #                         max_IoU_list=max_IoU_list, 
    #                         pixel_list=pixel_list,
    #                         oracle_iou=oracle_iou,
    #                         pixel_size=pixel_size,
    #                         savefolder='sample_imgs/inria_dg_center'
    #                         )

def visualize_dg_road(max_img=1000, num_cpu=50,move_img=False):
     # First lets read the center prompt result.csv
    df = pd.read_csv('result_IOUs/DG_road_center_object_wise_IOU.csv', 
                     index_col=0)
    
    scatter_arr = np.load('scatter_list/DG_road_scatter_list.npy')
    max_IoU_list, pixel_list = scatter_arr[:, 0], scatter_arr[:, 1]



    try: 
        pool = Pool(num_cpu)
        args_list = []
        for i in tqdm(range(min(len(df), max_img))):
            SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
                                                       df['prompt_ind'].values[i])
            oracle_iou = max_IoU_list[i]
            pixel_size = pixel_list[i]
            args_list.append(('SAM_output/DG_road_center_prompt_save',
                              SAM_pred_mask_name,
                              'datasets/DG_road/train/',
                              'datasets/DG_road/train/',
                              'test',
                              max_IoU_list,
                              pixel_list,
                              oracle_iou,
                              pixel_size,
                              'sample_imgs/dg_road_center',
                              move_img))
        pool.starmap(visualize_single_mask, args_list)
    finally:
        pool.close()
        pool.join()

            # for i in tqdm(range(min(len(df), max_img))):
    #     SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
    #                                                    df['prompt_ind'].values[i])
    #     oracle_iou = max_IoU_list[i]
    #     pixel_size = pixel_list[i]

        # visualize_single_mask(SAM_pred_folder='SAM_output/DG_road_center_prompt_save',
        #                     SAM_pred_mask_name=SAM_pred_mask_name,
        #                     gt_mask_folder='datasets/DG_road/train/',
        #                     img_folder='datasets/DG_road/train/',
        #                     save_prefix='test',
        #                     max_IoU_list=max_IoU_list, 
        #                     pixel_list=pixel_list,
        #                     oracle_iou=oracle_iou,
        #                     pixel_size=pixel_size,
        #                     savefolder='sample_imgs/dg_road_center'
        #                     )

def visualize_cloud(max_img=1000):
     # First lets read the center prompt result.csv
    df = pd.read_csv('result_IOUs/cloud_center_object_wise_IOU.csv', 
                     index_col=0)
    
    scatter_arr = np.load('scatter_list/cloud_scatter_list.npy')
    max_IoU_list, pixel_list = scatter_arr[:, 0], scatter_arr[:, 1]

    for i in tqdm(range(min(len(df), max_img))):
        if df['prompt_ind'].values[i] > 5:      # Only output 5 masks per image
            continue
        SAM_pred_mask_name = '{}_prompt_ind_{}'.format(df['img_name'].values[i].split('.')[0],
                                                       df['prompt_ind'].values[i])
        oracle_iou = max_IoU_list[i]
        pixel_size = pixel_list[i]

        visualize_single_mask(SAM_pred_folder='SAM_output/cloud_center_prompt_save',
                            SAM_pred_mask_name=SAM_pred_mask_name,
                            gt_mask_folder='datasets/cloud/train_processed',
                            img_folder='datasets/cloud/train_processed',
                            save_prefix='test',
                            max_IoU_list=max_IoU_list, 
                            pixel_list=pixel_list,
                            oracle_iou=oracle_iou,
                            pixel_size=pixel_size,
                            savefolder='sample_imgs/cloud_center'
                            )

if __name__ == '__main__':
    # visualize_solar()
    # visualize_inria_dg()
    # visualize_dg_road()
    # visualize_cloud()
    # visualize_inria_dg(move_img=True)
    visualize_dg_road(move_img=True)

    # visualize_single_mask(SAM_pred_folder='SAM_output/solar_pv_center_prompt_save',
    #                       SAM_pred_mask_name='11ska625755_23_13_prompt_ind_0__0.968635618686676_0.9178075790405273_0.7066956758499146.npy',
    #                       gt_mask_folder='datasets/solar_masks',
    #                       img_folder='datasets/solar-pv',
    #                       save_prefix='test',
    #                        scatter_npy='scatter_list/solar_pv_scatter_list.npy',
    #                       oracle_iou=0.5,
    #                       pixel_size=200,
    #                       )
    