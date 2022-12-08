import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import torchvision.transforms.functional as TF


class EquiposDataset(Dataset):
    def __init__(self, directory, labels, size, flip_chance=0.5, color_change_chance=0.10, gaussian_noise_chance=0.2, gaussian_noise_range=5.0, luminosity_changes_chance=0.125, transform=None):
        self.directory = directory
        self.img_files = listdir(directory)
        self.labels = labels

        self.transform = transform
        self.size = size
        self.flip_chance = flip_chance
        self.color_change_chance = color_change_chance
        self.gaussian_noise_chance = gaussian_noise_chance
        self.luminosity_changes_chance = luminosity_changes_chance
        self.gaussian_noise_range = gaussian_noise_range

    def __len__(self):
        return len(self.img_files)

    def getItem(self, index):
        filename = self.img_files[index]
        aux_Image = Image.open(f"{self.directory}/{filename}")
        aux_Image = resizeImg(aux_Image)
        image_Tensor = TF.to_tensor(aux_Image)
        return image_Tensor


def resizeImg(sourceImg):
    desired_width = 733
    desired_height = 565
    dim = (desired_width, desired_height)
    resized_img = cv2.resize(sourceImg, dsize=dim,
                             interpolation=cv2.INTER_AREA)
    return resized_img


def grabCut(img):
    img = resizeImg(img)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 733, 565)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    moddedImg = img*mask2[:, :, np.newaxis]
    return moddedImg
