# Machine Learning for the Galar Capsule Dataset Paper

Implementation for the reproduction of the experiments of the paper

**Galar - a large multi-label video capsule endoscopy dataset**

Download: [Figshare Dataset Link](https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616/1) 


## Preparation

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)

2. Open/Activate Anaconda

3. `conda create --name capsule --file requirements.yml`

4. `conda activate capsule`

5. `git clone https://github.com/EKFZ-AI-Endoscopy/GalarCapsuleML.git` and `cd GalarCapsuleML`


6. Download and extract images and splits: TODO



## Machine Learning

### Models

This repository enables you to train / fine-tune several deep learning models. Currently implemented are:

- resnet18
- resnet50
- resnet34
- resnet101
- resnet152
- vit_l_32
- vit_b_16
- efficientnet_v2_m
- efficientnet_v2_s

You can choose whether to used pre-trained weights (provided by pytorch pre-trained on ImageNet) and whether to freeze most of the model except of the last linear layer. (--freeze 1)

---

### Classification Problems:

It is possible to train / fine-tune

1. a single label (binary classification) 
    - any label like _blood_
2. multiple labels (which may occur independent of each other in one frame) (multilabel classification) 
    - _technical_multilabel_: bubbles and dirt
3. multiple classes (where there is exactle one class in each frame) (multiclass classification) 
    - _section_: mouth, esophagus, stomach, small intestines, colon
    - _technical_multiclass_: good view, reduced view, no view

---

### Dataset imbalance problem

As the distribution of frames per label is highly uneven this can be considered when doing classification tasks. 

Possible ways to handle the dataset imbalance are:

- Do nothing about it
- Weighted loss
- Weighted Sampling (including down- and upsampling)

---


### Necessary training parameters

|Parameter|Description|Example value|
|---|---|---|
|image_dir|Path where images are stored in the dataset.|/dataset/images|
|annotation_dir|Path where label csv files are stored.|/dataset/labels|
|output_dir|Path where to save results: Model weights, Confusion Matrices |/path/to/results|
|split_path|Path where the splits downloaded or generated are stored.|/path/to/splits|
|training_features|Classification Problem: Either a single label, or _section_, _technical_multiclass_, _technical_multilabel_|_section_|
|num_folds|k-fold cross-validation (here you can select k)|e.g. 2 or 5|
|folds|Which folds to run?|e.g. 0 (just first fold) or 0-1-2-3-4 (for all folds in 5-fold cross-validation)|


### Hyperparameters

The following parameters might be adjusted:

|Parameter|Description|Possible Values|
|---|---|---|
|batch_size|Number of frames processed per learning batch at once|e.g. 256 - depends on how many fit into the GPU memory; this value should not be too small especially when doing multi-label / multiclass training so that there are enough samples / images for each label in each batch|
|width / height|How to (down-)scale the images.|e.g. 256 / 256|
|epochs|How many epochs to train? Note, that early stopping is implemented using the last 5 values and a patience of 10 based on the validation loss value.|default: 100|
|learning_rate|Set learning rate. Depends on the optimizer chosen.|e.g. for adamW: 0.001|
|optimizer|Chosse optimizer. Note: Learning rate needs to be adjusted accordingly!|e.g. sgd / adam / adamW|
|model|Chosse model|default: resnet50; currently implemented: resnet18,resnet34,resnet50,resnet101,resnet152,vit_l_32,vit_b_16,efficientnet_v2_m,efficientnet_v2_s|
|pretrained|Choose whether to use pretrained weights (usually ImageNet)|0/1|
|freeze|Chosse whether to freeze (not train) parts of the network. Default implementation is to freeze anything but the last fully convolutional block. To modify this behaviour, refer to the model.py file. |0/1|
|step_size|For learning rate scheduler stepLR: After how many epochs to reduce the learning rate (multiply by gamma)?|e.g. 10|
|gamma|For learning rate scheduler stepLR: After each step_size number of epochs reduce the learning rate by how much (multiply with gamma)?|e.g. 0.9 -> lr=lr*0.9|
|dropout|A dropout layer is being added before the last linear layer of the classification head. Here you can set the dropout propability.|e.g. 0.5 must be between 0 and 1|
|weighted_loss|Enable weighted loss|0,1|
|weighted sampling|Enable weighted sampling|0,1|
|downsample_factor|Use downsample_factor > 1 to downsample the bigger classes relative to the smallest class|1 (no impact)|
|weights|Use custom class weights for weighted sampling or weighted loss. If disabled, automatically generated weights for balancing will be used.|e.g. 0.2,0.5,0.3|
|static_random_downsampling|Alternative imbalance mitigation technique: Randomly select some images for each class which will be used for all epochs. Uses __weights__ parameter. Choose x (number of images per class) with x=#images_of_class*weight[class] for each class in training and validation. The selection is consistent over all training epochs.|0,1|


### How to Run

Run the train.py file with the parameters described above.

The files _train_sections.py_, _train_single_label.py_, _train_technical_multilabel_ and _train_technical_multiclass_ provide example configurations and can be run with changes to the paths.


### Generate Splits

Splits will be stored in the following structure:

```
splits
    section
        split0
            train.csv
            val.csv
        split1
            train.csv
            val.csv
        split2
            train.csv
            val.csv
        test.csv
    blood
        split0
            train.csv
            val.csv
        split1
            train.csv
            val.csv
        test.csv
    ...
```

The train.csv, val.csv and test.csv are structured in the following way (according to the classification problem, the corresponding one-hot encoded labels are also included):

```csv
outside,mouth,esophagus,stomach,small intestine,colon,path
1,0,0,0,0,0,3/frame_000000.PNG
1,0,0,0,0,0,3/frame_000001.PNG
...
```

```
python gen_k_fold.py --label_dir /path/to/labels --split_dir /path/to/store/splits --problem section
```

|Parameter|Description|Possible Values|
|---|---|---|
|label_dir|Path of the video label csv files.|/path/to/labels|
|split_dir|Path of where to save the generated splits.|/path/to/store/splits|
|problem|Select classification problem: _section_, _technical_multiclass_, _technical_multilabel_ or any other single label.|e.g. _blood_ or _section_|


---


## Terms of Use

The dataset is made available for research and educational use without restrictions. For any other purposes, including competitions and commercial applications, obtaining written consent beforehand is required. Additionally, any documents or papers that utilize or mention the dataset, or that present experimental findings derived from the Kvasir-Capsule, must include a citation of the relevant article.

## Contact

Maxime Le Floch, Fabian Wolf, Lucian McIntyre, Christoph Weinert, Albrecht Palm, Konrad Volk, Paul Herzog, Sophie Helene Kirk, Jonas L. Steinhäuser, Catrein Stopp, Mark Enrik Geissler, Moritz Herzog, Stefan Sulk, Jakob Nikolas Kather, Alexander Meining, Alexander Hann, Jochen Hampe, Nora Herzog, and Franz Brinkmann.
Contact: maxime.lefloch@pm.me

## LICENCE

By licensing the dataset under a Creative Commons Attribution 4.0 International (CC BY 4.0) License which allows sharing, copying, and redistribution, as well as adaptation and transformation, we hope to advanceresearch in the field. For more details about Creative Commons licensing, please refer to https://creativecommons.org.

## Citation

Pending: Official citation will be created after the manuscript is accepted or published.

## Version

Version 1.0 - First Version (15.03.2024)
