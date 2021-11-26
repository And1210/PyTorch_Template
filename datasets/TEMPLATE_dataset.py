import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt

class TEMPLATEDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = os.path.join(configuration["dataset_path"], "{}".format(self._stage))

        #-----------------------------------------------------------------------
        #Here is where you can do things like preload data and labels or do image preprocessing


        #-----------------------------------------------------------------------


        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    #This function returns an data, label pair. All data processing and modification should be done by the end of this function
    def __getitem__(self, index):
        #Load data from xml, you will need to modify if annotations are in different format
        self._data = ET.parse(os.path.join(self.dataset_path, "annotations/road{}.xml".format(i)))
        filename = self._data.findall('./filename')[0].text

        #Image loading assuming the images are in the 'images' folder in the dataset root path
        pixels = Image.open(os.path.join(self.dataset_path, "images/{}".format(filename)))
        image = np.asarray(pixels)#.reshape(48, 48)
        image = image.astype(np.uint8)

        #Image resizing
        image = cv2.resize(cropped, self._image_size)

        #Image formatting
        image = np.dstack([image] * 1)

        #Some image augmentation
        image = seg(image=image)

        #Apply defined transforms to image from constructor (will convert to tensor)
        image = self._transform(image)
        #Ensure the target is set as the label
        target = label

        #image should be the image data, target should be the label
        return image, target

    def __len__(self):
        # return the size of the dataset, replace with len of labels array
        return len(os.listdir(os.path.join(self.dataset_path, 'annotations')))
