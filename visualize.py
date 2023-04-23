import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
from tqdm import tqdm

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
    alpha = 0.005 if len(pixel_list) > 1e5 else 0.05
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
                          savefolder='sample_imgs'
                          ):
    """
    Visualize a 1x3 
    """
    img_prefix = SAM_pred_mask_name.split('_prompt')[0]
    # Get original img
    img_re = os.path.join(img_folder, 
                          img_prefix.replace('mask','').replace('gt_patch','img_patch')+'*')
    img_name = glob.glob(img_re)[0]
    img = cv2.imread(img_name)
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
    if len(gt_mask_name) == 0:
        gt_mask = np.zeros_like(img)[:, :, 0]
    else:
        gt_mask_name = gt_mask_name[0]
    gt_mask = cv2.imread(gt_mask_name)

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
    
if __name__ == '__main__':
    visualize_solar()
    # visualize_single_mask(SAM_pred_folder='SAM_output/solar_pv_center_prompt_save',
    #                       SAM_pred_mask_name='11ska625755_23_13_prompt_ind_0__0.968635618686676_0.9178075790405273_0.7066956758499146.npy',
    #                       gt_mask_folder='datasets/solar_masks',
    #                       img_folder='datasets/solar-pv',
    #                       save_prefix='test',
    #                        scatter_npy='scatter_list/solar_pv_scatter_list.npy',
    #                       oracle_iou=0.5,
    #                       pixel_size=200,
    #                       )
    