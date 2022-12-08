import cv2
import numpy as np
import os
from os import listdir

def main():
    folder_dir = "training_dataset"
    height = []
    width = []
    for image in os.listdir(folder_dir):
        if(image.endswith(".jpg")):
            img = cv2.imread("training_dataset/"+ image)
            print(image)
            dimensions = img.shape
            height.append(img.shape[0])
            width.append(img.shape[1])
    print(f'height:   {sum(height) / len(height)}')
    print(f"width:   {sum(width) / len(width)}")

if __name__ == "__main__":
    main()
