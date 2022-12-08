
from torch import nn
import torch.nn.functional as F

class model_teamClassifer(nn.Module):
    def __init__(self, ratio_width=733, ratio_height=565, out=32):
        super(model_teamClassifer, self).__init__()
        self.ratio_height = ratio_height
        self.ratio_width = ratio_width
        self.out = out
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # linear layers
        self.linear1 = nn.Linear(512*self.ratio_width*self.ratio_height, 768)
        self.linear2 = nn.Linear(768, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, self.out)

        #droput
        self.dropout = nn.Dropout(p=0.1)
        