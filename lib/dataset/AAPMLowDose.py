#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/6/2019 3:12 PM

# sys
import os
import numpy as np
import SimpleITK as sitk
import pydicom
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch
import torch
import torch.utils.data as data

# monai
from monai.utils import MAX_SEED
from monai.transforms import (
    Randomizable,
    apply_transform,
    Compose,
    ResizeWithPadOrCropd,
    RandAffineD
)


class AAPMDataset(data.Dataset, Randomizable):
    def __init__(self,
                 data_root=r'./data',
                 data_list=r'train.txt',
                 transforms=None,
                 customized_dataset_size=0
                 ):
        super(AAPMDataset, self).__init__()

        self.data_root = data_root
        self.transforms = transforms

        self.sample_list = []
        with open(os.path.join(self.data_root, data_list), 'r') as f:
            self.sample_list = [x.strip() for x in f.readlines()]

        self._seed = 0
        self.real_dataset_size = len(self.sample_list)
        self.dataset_size = customized_dataset_size if customized_dataset_size else self.real_dataset_size

    def randomize(self):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, item):
        self.randomize()
        item = item % self.real_dataset_size
        noisy_path, clean_path = self.sample_list[item].split(' ')

        reader = sitk.ImageSeriesReader()

        # read noisy image
        noisy_dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(self.data_root, noisy_path))
        
        for i in noisy_dicom_names:
            if 'RD' in i:
                dose_scaling_noisy = pydicom.dcmread(i).DoseGridScaling # only for dose
                
        reader.SetFileNames(noisy_dicom_names)
        noisy_image = reader.Execute()

        # read clean image
        clean_dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(self.data_root, clean_path))
        
        for j in clean_dicom_names:
            if 'RD' in j:
                dose_scaling_clean = pydicom.dcmread(j).DoseGridScaling # only for dose
        
        reader.SetFileNames(clean_dicom_names)
        clean_image = reader.Execute()

        resolution = noisy_image.GetSpacing()

        noisy_image = sitk.GetArrayFromImage(noisy_image).astype(np.float32)
        noisy_image = torch.from_numpy(noisy_image[0]).unsqueeze(0)  # to handle Dose image input
        # noisy_image = torch.from_numpy(noisy_image).unsqueeze(0)  # to handle CT images input
        noisy_image = noisy_image * dose_scaling_noisy / 70.0  # divide dose prescription
        # noisy_image = (noisy_image + 1000) / 2000  # for CT images

        clean_image = sitk.GetArrayFromImage(clean_image).astype(np.float32)
        clean_image = torch.from_numpy(clean_image[0]).unsqueeze(0)  # to handle Dose image input
        # clean_image = torch.from_numpy(clean_image).unsqueeze(0)  # to handle CT images input
        clean_image = clean_image * dose_scaling_clean / 70.0  # divide dose prescription
        # clean_image = (clean_image + 1000) / 2000  # for CT images 

        result = apply_transform(self.transforms, data={'noisy_image': noisy_image, 'clean_image': clean_image})
        
        # TODO: the resolution should change accordingly after apply random transformation
        return {'input': result['noisy_image'],
                'target': result['clean_image'],
                'resolution': resolution,
                'path': noisy_path}

    def __len__(self):
        return self.dataset_size


def get_data_provider(cfg, phase='Train'):
    target_size = cfg.DATASET.TARGET_SIZE
    data_augmentation_probability = cfg.DATASET.DATA_AUGMENTATION_PROB
    rotation_angle = [i / 180 * np.pi for i in cfg.DATASET.ROTATION_DEGREE_ANGLE_ZYX]
    shear_range = cfg.DATASET.SHEAR_RANGE_ZYX
    scale_range = cfg.DATASET.SCALE_RANGE_ZYX
    translation_range = cfg.DATASET.TRANSLATION_RANGE_ZYX

    if phase.upper() == 'TRAIN':
        transform = Compose([RandAffineD(keys=['noisy_image', 'clean_image'],
                                           prob=data_augmentation_probability,
                                           rotate_range=rotation_angle,
                                           shear_range=shear_range,
                                           scale_range=scale_range,
                                           translate_range=translation_range,
                                           spatial_size=target_size,
                                           mode=['bilinear', 'bilinear'],
                                           padding_mode=['zeros', 'zeros'])])
    else:
        # transform = Compose([
        #     ResizeWithPadOrCropd(keys=['noisy_image', 'clean_image'],
        #                          spatial_size=target_size)
        # ])
        transform = Compose([RandAffineD(keys=['noisy_image', 'clean_image'],
                                           prob=0,
                                           rotate_range=0,
                                           shear_range=0,
                                           scale_range=0,
                                           translate_range=0,
                                           spatial_size=target_size,
                                           mode=['bilinear', 'bilinear'],
                                           padding_mode=['zeros', 'zeros'])])
        
    data_list = {'TRAIN': cfg.DATASET.TRAIN_LIST,
                 'VAL': cfg.DATASET.VAL_LIST,
                 'TEST': cfg.DATASET.TEST_LIST}

    batch_size = eval('cfg.{}.BATCHSIZE_PER_GPU'.format(phase.upper()))
    if torch.cuda.is_available():
        batch_size = batch_size * torch.cuda.device_count()

    iteration = int(cfg.TRAIN.TOTAL_ITERATION)
    current_dataset = AAPMDataset(data_root=cfg.DATASET.ROOT,
                                  data_list=data_list[phase.upper()],
                                  transforms = transform,
                                  customized_dataset_size=batch_size * iteration if phase.upper() == 'TRAIN' else 0
                                  )

    data_loader = torch.utils.data.DataLoader(current_dataset,
                                              batch_size=batch_size,
                                              shuffle=True if phase.upper() == 'TRAIN' else False,
                                              num_workers=cfg.WORKERS,
                                              pin_memory=torch.cuda.is_available())

    return data_loader


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    data_root = r'D:\data\1_Challenge\LowDoseChanllenge\Train\sharps'
    data_list = 'train.txt'
    is_train = True
    is_shuffle = True

    batch_size = 1
    num_threads = 0
    is_gpu = torch.cuda.is_available()

    train_transform = Compose([RandAffineD(keys=['noisy_image', 'clean_image'],
                                           prob=0.5,
                                           rotate_range=(9/180 * np.pi, 0, 0),
                                           shear_range=(0, 0, 0.1, 0.1, 0.1, 0.1),
                                           scale_range=(0.0, 0.2, 0.2),
                                           translate_range=(0, 32, 32),
                                           spatial_size=[-1, 256, 256],
                                           mode=['bilinear', 'bilinear'],
                                           padding_mode=['border', 'border']
    )])

    current_dataset = AAPMDataset(data_root=data_root,
                                  data_list=data_list,
                                  transforms=train_transform,
                                  customized_dataset_size=0)

    train_loader = torch.utils.data.DataLoader(current_dataset,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle,
                                               num_workers=num_threads,
                                               pin_memory=is_gpu)

    for i, data in enumerate(train_loader):
        current_image, current_target = data['input'], data['target']
        current_CT = current_image.squeeze().detach().cpu().numpy()
        current_target = current_target.squeeze().detach().cpu().numpy()

        idx = current_CT.shape[0] // 2
        print("iter {}, "
              "shape: {}, "
              "CT min/max: {}/{}, "
              "target min/max: {}/{}".format(i,
                                             current_CT.shape,
                                             np.min(current_CT), np.max(current_CT),
                                             np.min(current_target), np.max(current_target)))
        if True:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(current_CT[idx], cmap='gray', vmin=-250 / 2000 + 0.5, vmax=250 / 2000 + 0.5)
            plt.subplot(1, 3, 2)
            plt.imshow(current_target[idx], cmap='gray', vmin=-250 / 2000 + 0.5, vmax=250 / 2000 + 0.5)
            plt.subplot(1, 3, 3)
            plt.imshow(current_target[idx] - current_CT[idx], cmap='gray', vmin=-50 / 2000, vmax=50 / 2000)
            plt.show()

    print('Congrats! May the force be with you ...')
