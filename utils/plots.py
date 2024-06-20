import os
import cv2
import numpy as np
import torch



def plot_image_file(img_file):
    img = cv2.imread(img_file)
    output_file = os.path.join('outputs', os.path.basename(img_file))
    cv2.imwrite(output_file, img)

def make_color_label(label):
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8) # (H, W, 3) shape
    output_file = os.path.join('outputs', os.path.basename(label_file).replace('.png', '_label.png'))
    colors = [ # [B, G, R] value
        [0, 0, 0], # 0: background
        [144, 124, 226], # 1: motorway 
        [172, 192, 251], # 2: trunk
        [161, 215, 253], # 3: primary
        [187, 250, 246], # 4: secondary
        [255, 255, 255], # 5: tertiary
        [49, 238, 75], # 6: path
        [173, 173, 173], # 7: under construction
        [255, 85, 170], # 8: train guideway
        [234, 232, 120] # 9: airplay runway
    ]
    for i in range(10):
        color_label[label == i] = colors[i]
    cv2.imwrite(output_file, color_label)

    return color_label



def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    # if img is tensor, convert to cv2 image
    if torch.is_tensor(img):
        img = img.mul(255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    
    if label is not None:
        # if label_img is tensor, convert to cv2 image
        if torch.is_tensor(label):
            label = label.cpu().numpy().astype(np.uint8)
        color_label = make_color_label(label)
        label = color_label
        # overlay images
        img = cv2.addWeighted(img, 1.0, label, alpha, 0)
    # save image
    cv2.imwrite(save_file, img)

img_file = 'data/kari-road/val/images/BLD00072_PS3_K3A_NIA0276.png'
label_file = img_file.replace('images', 'labels')
img = cv2.imread(img_file)
label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE) # (H, W) shape
plot_image(img, label)

