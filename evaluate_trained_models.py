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
import shutil

def IoU_single_object_mask(gt_mask, pred_mask):
    """
    gt_mask and pred_mask are both 0/1 valued masks
    """
    intersection = gt_mask * pred_mask
    union = (gt_mask + pred_mask) > 0
    num_intersection = np.sum(intersection)
    num_union = np.sum(union)
    return  num_intersection / num_union, num_intersection, num_union

def evaluate_single_dataset(folder,):
    """
    Evaluate the pixel IOU of a single dataset
    """
    gt_mask_folder = os.path.join(folder, 'gt')
    pred_mask_folder = os.path.join(folder, 'masks')
    intersection_tot, union_tot = 0, 0
    for file in tqdm(os.listdir(gt_mask_folder)):
        gt_mask_file = os.path.join(gt_mask_folder, file)
        pred_mask = cv2.imread(os.path.join(pred_mask_folder, file))[:, :, 0]
        if not os.path.exists(gt_mask_file):
            intersection, union = 0, np.sum(pred_mask > 0)
        else:
            gt_mask = cv2.imread(gt_mask_file)[:, :, 0]
            _, intersection, union = IoU_single_object_mask(gt_mask, pred_mask)
        intersection_tot += intersection
        union_tot += union
    print('for dataset folder {}, IoU = {} (intersection {} pix, union {} pix)'.format(folder,
                                                                                       intersection_tot/union_tot
                                                                                       ,intersection_tot, 
                                                                                       union_tot))

def copy_file(pred_mask_dir, gt_source_dir, gt_target_dir):
    for file in tqdm(os.listdir(pred_mask_dir)):
        if '.csv' in file:
            continue
        source_file = os.path.join(gt_source_dir, file)
        tgt_file = os.path.join(gt_target_dir, file)
        if os.path.exists(source_file):
            shutil.copyfile(source_file, tgt_file)
    
def evaluate_multi_dataset():
    base_dir = '/home/sr365/SAM/detector_predictions'
    dataset_list = ['solar','inria_dg','cloud','dg_road']
    for dataset in dataset_list:
        evaluate_single_dataset(os.path.join(base_dir, dataset))

if __name__ == '__main__':
    evaluate_multi_dataset()
    
    # Copy files
    # copy_file(pred_mask_dir='/home/sr365/SAM/detector_predictions/solar/masks',
    #            gt_source_dir='/home/sr365/SAM/datasets/solar_masks', 
    #            gt_target_dir='/home/sr365/SAM/detector_predictions/solar/gt')


