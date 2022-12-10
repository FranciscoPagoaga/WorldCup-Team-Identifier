import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import torchvision.transforms.functional as TF


class EquiposDataset(Dataset):
    def __init__(self, directory, labels, size=(733, 565)):
        self.directory = directory
        self.img_files = listdir(directory)
        self.labels = labels
        self.size = size
        if self.labels != {}:
            classes = []
            for img in self.img_files:
                class_final = f"{self.labels[img]['equipo']}"
                self.labels[img]['class'] = class_final
                classes.append(class_final)
            self.classes = sorted(list(set(classes)))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        filename = self.img_files[index]
        #aux_Image = cv2.imread(f"{self.directory}/{filename}")
        aux_Image = Image.open(f"{self.directory}/{filename}")
        aux_Image = self.resizeImg(aux_Image)
        image_Tensor = TF.to_tensor(aux_Image)
        if self.labels != {}:
            class_final = self.labels[filename]['equipo']
            return image_Tensor, torch.tensor(self.classes.index(class_final))
        else:
            return image_Tensor, filename

    def resizeImg(self, sourceImg):
        # desired_width = 366
        # desired_height = 282
        # dim = (desired_width, desired_height)
        # resized_img = cv2.resize(sourceImg, dsize=dim,
        #                         interpolation=cv2.INTER_AREA)

        img_width, img_height = sourceImg.size
        target_width, target_height = self.size

        scale_w = target_width/img_width
        scale_h = target_height/img_height

        factor = 0
        if scale_h >= scale_w:
            factor = scale_w
            sourceImg = sourceImg.resize(
                (target_width, int(sourceImg.height * factor)))
            diff = (target_height - sourceImg.height)
            padding_top = diff // 2
            padding_bottom = diff - padding_top
            sourceImg = self.padding(
                sourceImg, padding_top, 0, padding_bottom, 0, (0, 0, 0))
        else:
            factor = scale_h
            sourceImg = sourceImg.resize(
                (int(sourceImg.width * factor), target_height))
            diff = (target_width - sourceImg.width)
            padding_right = diff // 2
            padding_left = diff - padding_right
            sourceImg = self.padding(
                sourceImg, 0, padding_right, 0, padding_left, (0, 0, 0))

        return sourceImg

    def grabCut(self, img):
        img = self.resizeImg(img)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, 733, 565)
        cv2.grabCut(img, mask, rect, bgdModel,
                    fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        moddedImg = img*mask2[:, :, np.newaxis]
        return moddedImg

    def padding(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
