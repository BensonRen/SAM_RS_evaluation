import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

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

image_name = 'crop_example/8904536.jpeg'
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
