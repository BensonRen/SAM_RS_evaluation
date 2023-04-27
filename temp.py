import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import pickle

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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, )
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size)

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

def visualize_single_img_and_prompt_with_SAM():
    # image = cv2.imread('solar-pv/11ska625740_31_05.tif')
    image = cv2.imread('datasets/crop/imgs/8904536.jpeg')
    
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


def visualize_single_img_and_its_point_prompt():
    # image = cv2.imread('solar-pv/11ska625740_31_05.tif')
    img_file = 'datasets/crop/imgs/8904536.jpeg'
    image = cv2.imread(img_file)
    mask = cv2.imread(img_file.replace('jpeg','png').replace('imgs','masks_filled'))
    prompt_file = 'point_prompt_pickles/crop_center_prompt.pickle'
    # Read the prompt
    with open(prompt_file, 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    # print(prompt_point_dict)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cur_point = prompt_point_dict[os.path.basename(img_file).replace('jpeg','png')]
    f = plt.figure(figsize=[10,5])
    ax = plt.subplot(121)
    plt.imshow(image)
    cur_point = np.stack(cur_point, axis=0)
    show_points(cur_point, np.ones(len(cur_point)), ax, marker_size=30)
    plt.axis('off')
    ax = plt.subplot(122)
    plt.imshow(mask)
    show_points(cur_point, np.ones(len(cur_point)), ax, marker_size=30)
    plt.axis('off')
    plt.savefig('investigation/crop_sample/test.png')

def check_multi_prompt_point_extraction_by_plotting():
    mode = 'multi_point_rand_50'
    with open('point_prompt_pickles/DG_road_{}_prompt.pickle'.format(mode), 'rb') as handle:
    # with open('point_prompt_pickles/inria_DG_{}_prompt.pickle'.format(mode), 'rb') as handle:
        prompt_point_dict = pickle.load(handle)
    # folder = 'datasets/Combined_Inria_DeepGlobe_650/patches'
    folder = 'datasets/DG_road/train'
    # print(prompt_point_dict)
    for key in list(prompt_point_dict.keys()):

        # print(type(prompt_point_dict[key]))
        # print(len(prompt_point_dict[key]))
        # for i in range(len(prompt_point_dict[key])):
        #     print(np.shape(prompt_point_dict[key][i]))
        # print(prompt_point_dict[key])
        # quit()
        
        # Read the image
        img = cv2.imread(os.path.join(folder, key.replace('mask','sat').replace('png','jpg')))
        f = plt.figure()
        plt.imshow(img)
        coords = prompt_point_dict[key][0]
        print(np.shape(coords))
        labels = np.ones(len(coords)).astype('int')
        show_points(coords, labels, plt.gca(), marker_size=30)
        plt.savefig('investigation/multi_point_prompt/test.png')
        # for point in prompt_point_dict[key]:
            
        quit()

if __name__ == '__main__':
    visualize_single_img_and_its_point_prompt()
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

    # visualize_single_img_and_prompt()

    check_multi_prompt_point_extraction_by_plotting()

