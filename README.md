# FeatureKD: "Feature Based Knowledge Distillation for Fall Detection"
Implementation of FeatureKD proposed in paper titled "SmartFallMM: A Multimodal Dataset For Enhanced Fall Detection" 

## Getting started 
- Create an pip environment and use the requirements.txt to install all the neccasary files.

```bash
pip install -r requirements.txt
```
This requirements file doesn't have the instructions to install pytorch. Please install pytorch 1.13.0 for the experiments

## Get the Dataset
- Download the SmartFallMM data from [this link](https://sites.google.com/view/smartfall-mm/home). Put the dataset under `data` folder. 


## Choose and configure models
- Model configuration for InertialTransformer model is kept under ``config/smartfallmm/student.yaml``.

- Model Configuration or BiScaleFormer is kept under ``config/smartfallmm/teacher.yaml`` for SmartFallMM dataset.

- Configuration for Distillation is stored in ``config/smartfallmm/distill.yaml`` for SmartFallMM dataset.

## Step for Knowledge Distillation 
- Train a teacher model first using ``main.py`` and ``config/smartfallmm/teacher.yaml``. You can do it by uncommenting `line 57` in `train.sh` . 
- Perform knowledge distillation using ``distill.py`` and ``config/smartfallmm/distill.yaml`` . You can perform the distillation by uncomenting `line 61`.

## Train and test
Give execution access to ``train.sh`` with 
```bash
chmod +x ./train.sh
```
Run the ``train.sh`` to train and test the multimodal and accelerometer models. Log and weights would be saved under working directory. Use the following command to run the ``train.sh`` script.

```bash
./train.sh
```


