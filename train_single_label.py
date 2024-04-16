# Fine-tuning for single labels

import os

optimizer = "adamW"
model = "resnet50"
epochs = 100
batch_size = 128

image_size = 224

learning_rate = 0.001

# Which folds to train on (separated by -)
train_folds = "0-1"

# Select the label here
training_features = ['blood'] # ['pylorus', 'ileocecal valve', 'ulcer', 'polyp', 'active bleeding', 'blood', 'erythema', 'erosion', 'angiectasia'] 

options = {
    "image_dir": "/path/to/images",
    "annotation_dir": "/path/to/labels",
    "output_dir": "/path/to/results_dir",
    "splits_path": "/path/to/splits", # this path should just point to the general split directory, not to the splits for the specific label!
}

for feature in training_features:
    _ = os.system(f'python train.py --image_dir {options["image_dir"]} --annotation_dir {options["annotation_dir"]} --output_dir {options["output_dir"]} --split_path {options["splits_path"]} --epochs {epochs} --training_features {feature} --learning_rate {learning_rate} --optimizer {optimizer} --model {model} --pretrained 1 --freeze 0 --num_folds 2 --folds {train_folds} --batch_size {batch_size} --height {image_size} --width {image_size} --weighted_loss 1') 

# The --dual_output option uses two model outputs for a binary classification (one class per logit), while without this option only one output is thresholded by 0.5
