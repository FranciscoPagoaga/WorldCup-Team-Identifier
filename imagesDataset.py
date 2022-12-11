import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from os import listdir
from PIL import Image
import torchvision.transforms.functional as TF


class EquiposDataset(Dataset):
    def __init__(self, directory, labels, size=(512, 512)):
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
            real_out = [0.0]*32
            index = self.classes.index(class_final)
            real_out[index] = 1.0
            return image_Tensor, torch.tensor(real_out)
        else:
            return image_Tensor, filename

    def resizeImg(self, sourceImg):
        sourceImg = sourceImg.resize(self.size)


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
