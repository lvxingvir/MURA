from __future__ import print_function, division
import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
from common import config
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class MURA_Dataset(Dataset):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, data_dir, csv_file, transform=None):
        """
        :param data_dir: the directory of data
        :param csv_file: the .csv file of data list
        :param transform: the transforms exptected to be applied to the data
        """
        self.data_dir = data_dir
        self.frame = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform

    def _parse_patient(self, img_filename):
        """
        :param img_filename: the file name of the image data
        :return: the number of patient
        """
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        """
        :param img_filename: the file name of the image data
        :return: the number of study
        """
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        """
        :param img_filename: the file name of image data
        :return: the number of image
        """
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        """
        :param img_filename: the file name of image data
        :return: the type of the study
        """
        return self._study_type_re.search(img_filename).group(1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_filename = self.frame.iloc[idx, 1]
        # print(idx,img_filename)
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        file_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(file_path).convert('RGB')
        label = self.frame.iloc[idx, 2]
        label = int(label)

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'file_path': file_path,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }

        if self.transform:
            image = self.transform(image)

        # plt.imshow(image.permute(1,2,0).numpy())
        sample = {'image': image, 'label': label, 'meta_data': meta_data}
        return sample


def get_dataloaders(name, batch_size, shuffle, num_workers=32, data_dir=config.data_dir):
    """
    :param name: the phase for transforms
    :param batch_size: the size of an batch
    :param shuffle: whether random shuffle the data or not
    :param num_workers: the number of workers
    :return: the dataloader
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(448),
            # transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            # transforms.ColorJitter(),   # to do
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    image_dataset = MURA_Dataset(data_dir=data_dir, csv_file='MURA-v1.1/%s.csv' % name,
                                 transform=data_transforms[name])
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def calc_data_weights():
    """
    :return: the weights of positive and negative data of each type of study
    """
    frame = pd.read_csv(r'E:\Xing\Data\MURA-v1.1\MURA-v1.1\\train.csv')
    n_t = {t: 0 for t in config.study_type}
    a_t = {t: 0 for t in config.study_type}
    w_t0 = {t: 0.0 for t in config.study_type}
    w_t1 = {t: 0.0 for t in config.study_type}

    study_type_re = re.compile(r'XR_(\w+)')

    for idx in range(len(frame)):
        img_filename = frame.iloc[idx, 1]
        study_type = study_type_re.search(img_filename).group(1)

        label = frame.iloc[idx, 2]
        label = int(label)
        if label == 1:
            a_t[study_type] += 1
        else:
            n_t[study_type] += 1

    for t in config.study_type:
        w_t0[t] = a_t[t] / (a_t[t] + n_t[t])
        w_t1[t] = n_t[t] / (a_t[t] + n_t[t])

    return [w_t0, w_t1]


if __name__ == "__main__":

    LOSS_WEIGHTS = calc_data_weights()

    train_loader = get_dataloaders('train', batch_size=1,
                                   shuffle=False)

    valid_loader = get_dataloaders('valid', batch_size=1,
                                   shuffle=False)

    for i, data in enumerate(tqdm(train_loader)):
        # print(data['meta_data']['img_filename'])
        inputs = data['image']
        labels = data['label']
        study_type = data['meta_data']['study_type']
        file_paths = data['meta_data']['file_path']
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [LOSS_WEIGHTS[labels[i]][study_type[i]] for i in range(inputs.size(0))]

        # print(weights)