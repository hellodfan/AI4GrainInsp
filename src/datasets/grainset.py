import os
from enum import Enum

import random
import glob
import PIL
import torch
import cv2
import numpy as np
from torchvision import transforms
from datasets.aug import get_aug_img
from albumentations import (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
                            Resize, CenterCrop, Compose, Normalize)
from albumentations.pytorch import ToTensorV2

_CLASSNAMES = [
    "grain"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def ImageList(path):
    imgs, labels = [], []
    count = 0
    root = '/'.join(path.split('/')[:-1])
    with open(path, "r") as f:
        content = f.readlines()

    random.shuffle(content)

    for i in content:
        line = i.rstrip()
        words = line.split()
        im_path = os.path.join(root, words[0])
        if os.path.exists(im_path):
            imgs.append(im_path)
            labels.append(int(words[1]))
        else:
            # print(im_path)
            count += 1
    
    print('not valid:',count)
    print('valid count:',len(imgs))

    return imgs, labels


class GrainDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Grain.
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
        self.img_paths, self.labels = ImageList(source)
        self.resize_shape = (imagesize, imagesize)

        print('### cur labels sum:',sum(self.labels))

        self.transform_img = Compose([
            # RandomBrightnessContrast(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

        self.imagesize = (3, imagesize, imagesize)

        self.anomaly_source_paths = sorted(glob.glob('./texture/grass/*.jpg'))


    def flip(self, imgs, axis, p=0.5):
        if random.random()<p:
            for ix in range(len(imgs)):
                if imgs[ix] is not None:
                    imgs[ix] = np.flip(imgs[ix],axis).copy()
        return imgs


    def __getitem__(self, idx):
        # classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image_path = self.img_paths[idx]
        combine_image = cv2.imread(image_path)
        im_h, im_w, im_c = combine_image.shape
        image = combine_image[:,:im_w//2,:]
        shape_mask  = combine_image[:,-im_w//2:,:]

        # image, shape_mask = self.flip([image, shape_mask], axis=0)
        # image, shape_mask = self.flip([image, shape_mask], axis=1)

        image, aug_image, shape_mask, perlin_mask = get_aug_img(image, shape_mask, resize_shape=self.resize_shape, texture_path=random.choice(self.anomaly_source_paths))

        image = self.transform_img(image=image)['image']
        aug_image = self.transform_img(image=aug_image)['image']
        shape_mask  = torch.from_numpy(shape_mask)
        perlin_mask = torch.from_numpy(perlin_mask)
        shape_mask  = (shape_mask>0.1).int()
        perlin_mask = (perlin_mask>0.1).int()

        if self.split!=DatasetSplit.TEST:
            mask = perlin_mask
        else:
            mask = shape_mask

        anomaly = 'good' if self.labels[idx]==0 else 'bad'

        return {
            "image": image,
            "aug_image": aug_image,
            "mask": mask,
            "classname": 'grain',
            "anomaly": anomaly,
            "is_anomaly": self.labels[idx],
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.img_paths)

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
