# Run training for the labels bubbles and dirt (multilabel).

import os

optimizer = "adamW"
model = "resnet50"
epochs = 100
batch_size = 128

image_size = 224

learning_rate = 0.001

# Which folds to train on (separated by -)
train_folds = "0-1-2-3-4"

options = {
    "image_dir": "/path/to/images",
    "annotation_dir": "/path/to/labels",
    "output_dir": "/path/to/results_dir",
    "splits_path": "/path/to/splits", # this path should just point to the general split directory, not to the splits for the specific label!
}

# Standard fine-tuning

_ = os.system(
    f"python train.py --image_dir {options['image_dir']} --annotation_dir {options['annotation_dir']} --output_dir {options['output_dir']} --split_path {options['splits_path']} --epochs {epochs} --training_features technical_multilabel --downsample_factor 1 --learning_rate {learning_rate} --optimizer {optimizer} --model {model} --pretrained 1 --freeze 0 --num_folds 5 --folds {train_folds} --batch_size {batch_size} --height {image_size} --width {image_size} --dropout 0.2"
)

# Fine-tuning with weighted loss

_ = os.system(
    f"python train.py --image_dir {options['image_dir']} --annotation_dir {options['annotation_dir']} --output_dir {options['output_dir']} --split_path {options['splits_path']} --epochs {epochs} --training_features technical_multilabel --downsample_factor 1 --learning_rate {learning_rate} --optimizer {optimizer} --model {model} --pretrained 1 --freeze 0 --num_folds 5 --folds {train_folds} --batch_size {batch_size} --height {image_size} --width {image_size} --weighted_loss 1 --dropout 0.2"
)

# Fine-tuning with weighted sampling
# Use downsample_factor > 1 to downsample the bigger classes relative to the smallest class

_ = os.system(
    f"python train.py --image_dir {options['image_dir']} --annotation_dir {options['annotation_dir']} --output_dir {options['output_dir']} --split_path {options['splits_path']} --epochs {epochs} --training_features technical_multilabel --downsample_factor 4 --learning_rate {learning_rate} --optimizer {optimizer} --model {model} --pretrained 1 --freeze 0 --num_folds 5 --folds {train_folds} --batch_size {batch_size} --height {image_size} --width {image_size} --weighted_sampling 1 --dropout 0.2"
)

