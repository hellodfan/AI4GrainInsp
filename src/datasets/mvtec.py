import os
from enum import Enum

import glob
import random
import cv2
import PIL
import torch
import numpy as np
from torchvision import transforms
from datasets.aug import get_aug_img
from albumentations import (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
                            Resize, CenterCrop, Compose, Normalize)
from albumentations.pytorch import ToTensorV2

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = Compose([
            # RandomBrightnessContrast(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

        self.imagesize = (3, imagesize, imagesize)
        self.resize_shape = (imagesize, imagesize)

        self.center_crop = Compose([
            Resize(resize, resize),
            CenterCrop(imagesize, imagesize),
        ])

        self.anomaly_source_paths = sorted(glob.glob('./texture/dtd/images/*/*.jpg'))
        

    
    def flip(self, imgs, axis, p=0.5):
        if random.random()<p:
            for ix in range(len(imgs)):
                if imgs[ix] is not None:
                    imgs[ix] = np.flip(imgs[ix],axis).copy()
        return imgs


    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = cv2.imread(image_path)
        im_h, im_w, im_c = image.shape
        image = self.center_crop(image=image)['image']
        shape_mask = np.ones((im_h, im_w, im_c), dtype=np.uint8)*255

        test_mask = None
        if mask_path is not None:
            test_mask = cv2.imread(mask_path)
            test_mask = self.center_crop(image=test_mask)['image']
        
        image, shape_mask, test_mask = self.flip([image, shape_mask, test_mask], axis=0)
        image, shape_mask, test_mask = self.flip([image, shape_mask, test_mask], axis=1)

        image, aug_image, shape_mask, perlin_mask = get_aug_img(image, shape_mask, resize_shape=self.resize_shape, texture_path=random.choice(self.anomaly_source_paths))

        image = self.transform_img(image=image)['image']
        aug_image = self.transform_img(image=aug_image)['image']
        shape_mask  = torch.from_numpy(shape_mask)
        perlin_mask = torch.from_numpy(perlin_mask)
        shape_mask  = (shape_mask>0.1).int()
        perlin_mask = (perlin_mask>0.1).int()

        if self.split == DatasetSplit.TEST and mask_path is not None:
            test_mask  = cv2.resize(test_mask[:,:,0], self.resize_shape, interpolation=cv2.INTER_NEAREST)
            test_mask = torch.from_numpy(test_mask)
            test_mask  = (test_mask>0.1).int()
            mask = test_mask
        else:
            mask = perlin_mask

        return {
            "image": image,
            "aug_image": aug_image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate