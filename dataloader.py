import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import ClassificationType
import collections
from PIL import Image

from sklearn.preprocessing import LabelEncoder

class DataSetPreparationExternalSplit():
    """Dataset preparation using external split files (train.csv, val.csv, test.csv)"""

    def __init__(self, classification_type: ClassificationType, log, fold:int = 0, args=None) -> None:
        self.classification_type = classification_type
        self.log = log

        self.args = args

        self.downsample_factor = args.downsample_factor

        self.split_training_path = os.path.join(args.split_path, args.training_features, f'split_{fold}', 'train.csv')
        self.split_validation_path = os.path.join(args.split_path, args.training_features, f'split_{fold}', 'val.csv')

        self.test_path = os.path.join(args.split_path, args.training_features, 'test.csv')

        self.training_transform = A.Compose([
            A.Resize(args.width,args.height),
            #A.RandomCrop(height=224, width=224, p=1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.validation_transform = A.Compose([
            A.Resize(args.width,args.height),
            #A.CenterCrop(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def prepare_dataset_global(self, args, features:list, filter_unknown=True) -> list[Dataset]:
        all_datasets = [EndoCapsuleDataExternalSplit(self.split_training_path, args.width, args.height, features, self.classification_type, args, transform=self.training_transform, filter_unknown=filter_unknown), 
                        EndoCapsuleDataExternalSplit(self.split_validation_path, args.width, args.height, features, self.classification_type, args, transform=self.validation_transform, filter_unknown=filter_unknown),
                        EndoCapsuleDataExternalSplit(self.test_path, args.width, args.height, features, self.classification_type, args, transform=self.validation_transform, filter_unknown=filter_unknown, test_set=True)]

        return all_datasets

    def prepare_dataset(self, args, features: list[str]) -> tuple[DataLoader, DataLoader, DataLoader]:

        training_dataset = EndoCapsuleDataExternalSplit(self.split_training_path, args.width, args.height, features, self.classification_type, args, transform=self.training_transform)
        validation_dataset = EndoCapsuleDataExternalSplit(self.split_validation_path, args.width, args.height, features, self.classification_type, args, transform=self.validation_transform)
        test_dataset = EndoCapsuleDataExternalSplit(self.test_path, args.width, args.height, features, self.classification_type, args, transform=self.validation_transform, test_set=True)

        name = 'train_' + '_'.join(features).replace(' ', '-')

        if args.weighted_sampling == 1:
            weighted_sampler, self.train_weight = self.get_weighted_sampler(training_dataset, len(features), name)
        else:
            weighted_sampler = None

        # remove shuffle option, because WeightedRandomSampler will take care of that
        train_loader = DataLoader(training_dataset, shuffle=True if args.weighted_sampling == 0 else False, batch_size=args.batch_size, num_workers=args.j,pin_memory=args.pin_memory, sampler= weighted_sampler)

        name = 'val_' + '_'.join(features).replace(' ', '-')
        weighted_sampler, self.val_weight = self.get_weighted_sampler(validation_dataset, len(features), name)
        val_loader = DataLoader(validation_dataset, batch_size=int(args.batch_size * args.batch_size_validation_factor), shuffle=False, num_workers=args.j,pin_memory=args.pin_memory)

        name = 'test_' + '_'.join(features).replace(' ', '-')
        test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size * args.batch_size_validation_factor), shuffle=False, num_workers=args.j,pin_memory=args.pin_memory)

        return train_loader,val_loader, test_loader

    def get_label_combination(self, y):
        # group all combinations of labels for multi-label problem
        # This is just a very simple and non-optimal approach for balancing the labels in a multi-label classification
        return LabelEncoder().fit_transform([''.join(str(l)) for l in y])

    def get_weighted_sampler(self, dataset: Dataset, n_classes: int, name: str, use_cache:bool=False) -> tuple[WeightedRandomSampler, np.ndarray]:
        """Generate weights for weighted sampling and returns weighted sampler and weights."""

        if not os.path.exists('sample_weights'):
            os.makedirs('sample_weights')
        filename = 'sample_weights/{}.weights'.format(name)

        if os.path.isfile(filename) and use_cache:
            self.log('{}: load weights from cache'.format(name))
            samples_weight = torch.load(filename)
            
        else:
            # print('-----------------ELSE--------------------')
            if use_cache:
                self.log('{}: no cached weights found'.format(name))
            else:
                self.log('{}: calculate temporary weights'.format(name))

            targets = dataset.get_targets()

            #for ds in dataset.datasets:
            #    targets.extend(ds.get_targets()) 

            targets = torch.tensor(targets)

            class_sample_count = torch.sum(targets, dim=0)

            if n_classes == 1:
                tuple = (class_sample_count, torch.tensor([len(targets)-class_sample_count[0]]))
                class_sample_count = torch.cat(tuple,dim=0)

            self.log('classwise occurrences: {}'.format(class_sample_count))

            weight = 1. / class_sample_count.float()

            weight = F.normalize(weight, dim=0).detach().cpu().numpy()
            
            if self.args.weights != '':
                weight = [float(w) for w in self.args.weights.split(',')]
                self.log(f'USE CUSTOM WEIGHTS: {weight}')
            else:
                self.log(f'USE CALCULATED WEIGHTS: {weight}')

            if self.classification_type == ClassificationType.MULTILABEL_TECHNICAL:
                samples_combinations = self.get_label_combination(targets)

                counter = collections.Counter(samples_combinations)
                self.log(f'label combination counts: {counter}')

                counter_dict = dict(counter)

                weight_comb = []

                for i in range(0,len(counter_dict)):
                    weight_comb.append(counter_dict[i])

                weight_comb = 1. / torch.tensor(weight_comb)

                weight_comb = F.normalize(weight_comb, dim=0)

                samples_weight = [weight_comb[i] for i in samples_combinations]

                samples_weight = np.asarray(samples_weight)
                samples_weight = torch.from_numpy(samples_weight)

                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), int(len(samples_weight)/self.downsample_factor), replacement=True)
                return sampler, weight

            # print(f'TARGETS: {targets}')

            samples_weight = []
            for index, target in enumerate(targets):
                if n_classes == 1:
                    if target[0] == 1:
                        samples_weight.append(weight[0])
                    else:
                        samples_weight.append(weight[1])
                else:
                    for sample_class, value in enumerate(target):
                        if value == 1:
                            if not index < len(samples_weight):
                                samples_weight.append(weight[sample_class])
                            elif samples_weight[index] < weight[sample_class]:
                                # always select highest weight for each sample (this behaviour might not be what we want in some multi-label situations ...)
                                samples_weight[index] = weight[sample_class]
                        else:
                            if not index < len(samples_weight):
                                samples_weight.append(0)

            if use_cache:
                torch.save(samples_weight, filename)

        samples_weight = np.asarray(samples_weight)
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), int(len(samples_weight)/self.downsample_factor), replacement=True)
        return sampler, weight


class EndoCapsuleDataExternalSplit(Dataset):
    def __init__(
        self, split_path, width, height, features, classification_type, args, transform=None, filter_unknown=True, test_set=False):
        self.transform = transform
        self.images_features = []
        self.split_path = split_path
        self.width = width
        self.height = height
        self.features = features
        self.classification_type = classification_type
        self.args = args

        # filter all frames with unknown != 0
        df = pd.read_csv(split_path, dtype={'unknown': 'str'})

        self.df = df

        if args.static_random_downsampling and not test_set:
            self.df = None
            for i, weight in enumerate([float(weight) for weight in args.weights.split(',')]):
                df_part = df[df.iloc[:,i] == 1]
                df_filtered = df_part.sample(frac=weight, replace=True, random_state=42)

                if self.df is None:
                    self.df = df_filtered
                else:
                    self.df = pd.concat([self.df, df_filtered])

            self.df.drop_duplicates(subset=['path'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.images_features = self.df[self.features].values.tolist()
                                
    def load_image(self, image_name):
        try:
            im = Image.open(image_name)
            if im.mode != 'RGB':
                im = im.convert(mode='RGB')

        except:
            print(f'IMAGE FAILED TO LOAD: {image_name}')
            exit()
        return np.asarray(im)

    def __getitem__(self, index):

        labels = torch.tensor(self.images_features[index])

        image_path = os.path.join(self.args.image_dir, os.path.normpath(str(self.df.iloc[[index]]['path'].item())))

        image = self.load_image(image_path)

        if self.transform is not None:
            image = self.transform(image=image)

        return image['image'], labels

    def __len__(self) -> int:
        return len(self.images_features)

    def get_targets(self):
        return self.images_features
