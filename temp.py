import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

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

def visualize_SAM_logits_etc():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)


    image_name = '11ska625740_31_05_original_img.tif'
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_point = np.array([[139, 203]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks_2, scores_2, logits_2 = predictor.predict(
        mask_input = logits[None, 0, :, :],
        multimask_output=True,
    )

    # Plotting the SAM logits
    for i in range(3):
        f = plt.figure(figsize=[15, 5])
        plt.tight_layout()
        plt.subplot(221)
        plt.imshow(logits[i, :, :])
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(logits_2[i, :, :])
        plt.colorbar()
        plt.subplot(223)
        plt.hist(np.ravel(logits[i, :, :]))
        plt.subplot(224)
        plt.hist(np.ravel(logits_2[i, :, :]))
        plt.savefig(os.path.join('/home/sr365/SAM/investigation/logit_distribution', 
                                    os.path.basename(image_name).replace('.','_mask_{}.'.format(i))))

    # Plotting the SAM output mask for first round
    f = plt.figure(figsize=[15, 5])
    for i in range(3):
        plt.subplot(int('13{}'.format(i+1)))
        plt.imshow(image)
        show_mask(masks[i, :, :], plt.gca())
        plt.title(f"Mask {i+1}, Score: {scores[i]:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join('/home/sr365/SAM/investigation/logit_distribution', 
                                os.path.basename(image_name).replace('.','_mask_original_tot')))

    # Plotting the SAM output mask for second round
    f = plt.figure(figsize=[15, 5])
    for i in range(3):
        plt.subplot(int('13{}'.format(i+1)))
        plt.imshow(image)
        show_mask(masks_2[i, :, :], plt.gca())
        plt.title(f"Mask {i+1}, Score: {scores_2[i]:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join('/home/sr365/SAM/investigation/logit_distribution', 
                                os.path.basename(image_name).replace('.','_mask_2_tot')))

def visualize_prompt(mask_prop_dict):
    from prompt_solar import make_prompt_mask
    mask_file = '11ska625740_31_05.tif'
    mask = cv2.imread(mask_file)[:, :, 0]
    mask_output = make_prompt_mask(mask/255, 
                                   mask_prop_dict=mask_prop_dict)

    save_dir = '/home/sr365/SAM/investigation/prompt_shape'
    # Visualize the output
    f = plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.imshow(mask)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(mask_output)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, os.path.basename(mask_file).replace('.tif', '.png')))
    return mask_output

def evaluate_with_mask(input_mask, image_name, gt_mask_name, save_name=''):
    if len(np.shape(input_mask)) == 2:
        print('shape of input mask is ', np.shape(input_mask))
        input_mask = input_mask[None, :, :]
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    gt_mask = cv2.imread(gt_mask_name)[:, :, 0]

    masks, scores, logits = predictor.predict(
        mask_input = input_mask,
        multimask_output=True,
    )

    f = plt.figure(figsize=[20, 20])
    plt.subplot(331)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(332)
    plt.imshow(gt_mask)
    plt.axis('off')
    plt.subplot(333)
    plt.imshow(input_mask[0, :, :])
    plt.axis('off')
    plt.colorbar()
    for i in range(3):
        plt.subplot(int('33{}'.format(i+4)))
        plt.imshow(image)
        show_mask(masks[i, :, :], plt.gca())
        plt.axis('off')
    for i in range(3):
        plt.subplot(int('33{}'.format(i+7)))
        plt.imshow(logits[i, :, :])
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('investigation/prompt_engineering',
                os.path.basename(image_name).replace('.',save_name+'.')))

    # # Plotting the SAM logits
    # for i in range(3):
    #     f = plt.figure(figsize=[15, 5])
    #     plt.tight_layout()
    #     plt.subplot(221)
    #     plt.imshow(logits[i, :, :])
    #     plt.colorbar()
    #     # plt.subplot(222)
    #     # plt.imshow(logits_2[i, :, :])
    #     # plt.colorbar()
    #     plt.subplot(223)
    #     plt.hist(np.ravel(logits[i, :, :]))
    #     # plt.subplot(224)
    #     # plt.hist(np.ravel(logits_2[i, :, :]))
    #     plt.savefig(os.path.join('/home/sr365/SAM/investigation/logit_distribution', 
    #                                 os.path.basename(image_name).replace('.','_mask_{}.'.format(i))))

    # # Plotting the SAM output mask for first round
    # f = plt.figure(figsize=[15, 5])
    # for i in range(3):
    #     plt.subplot(int('13{}'.format(i+1)))
    #     plt.imshow(image)
    #     show_mask(masks[i, :, :], plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {scores[i]:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()
    # plt.tight_layout()
    # plt.savefig(os.path.join('/home/sr365/SAM/investigation/logit_distribution', 
    #                             os.path.basename(image_name).replace('.','_mask_original_tot')))

def visualize_single_img_and_prompt():
    image = cv2.imread('solar-pv/11ska625740_31_05.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point = np.array([[140, 25]])
    input_label = np.array([1])

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    # plt.show() 
    plt.savefig('investigation/exp_design_illustration/point_with_img.png')

    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()
        plt.savefig('investigation/exp_design_illustration/point_prompt_masks_{}.png'.format(i))

    input_box = np.array([127, 16, 158, 39])
    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    # plt.show()
    plt.savefig('investigation/exp_design_illustration/box_prompted_mask.png')


if __name__ == '__main__':
    # Testing different prompt shapes
    # mask_prop_dict = {'choice':'kernel', 
    #                     'mag':10,
    #                     'background': -10,
    #                     'kernel_size':10}
    # mask = visualize_prompt(mask_prop_dict)
    # evaluate_with_mask(mask, image_name='11ska625740_31_05_original_img.tif', 
    #                    gt_mask_name='11ska625740_31_05.tif',
    #                    save_name='{}_mag_{}_bg_{}_ks_{}'.format(mask_prop_dict['choice']))
    
    
    
    #visualize_SAM_logits_etc()

    visualize_single_img_and_prompt()

