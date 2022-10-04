# 3rd place Solution for [It's Corn (PogChamps #3)](https://www.kaggle.com/competitions/kaggle-pog-series-s01e03/leaderboard)

| Experiment    | Pretrained Model                    | img_size | int_lr | bs  | wd    | epochs | CV score | ensemble weight |
| ------------- | ----------------------------------- | -------- | ------ | --- | ----- | ------ | -------- | --------------- |
| NB_EXP_V2_007 | beit_large_patch16_224              | 224      | 2e-05  | 16  | 1e-08 | 16     | 0.82682  | 0.96109         |
| NB_EXP_V2_002 | beit_base_patch16_224               | 224      | 2e-05  | 16  | 1e-08 | 16     | 0.81846  | 0.41927         |
| NB_EXP_V2_008 | swin_base_patch4_window12_384_in22k | 384      | 6e-05  | 16  | 0.05  | 16     | 0.81672  | 0.52334         |
| NB_EXP_V2_006 | convnext_large_in22ft1k_224         | 224      | 1e-04  | 16  | 0.05  | 5      | 0.81658  | 0.79245         |
| NB_EXP_V2_005 | swin_large_patch4_window7_224       | 224      | 6e-05  | 16  | 0.01  | 16     | 0.81630  | 0.891163        |

The final inference notebook in available [here](https://www.kaggle.com/code/benihime91/pg3-corn-ensemble-submission-best-cv/notebook?scriptVersionId=107125251) and the weights of models in ensemble were determined by this [notebook](https://www.kaggle.com/code/benihime91/fork-of-pg3-corn-ensemble-find-weights/notebook).

I used `AdamW` optimizer with `CosineAnnealingLR` lr_scheduler for training. For loss i used simple `CrossEntropyLoss`.

Augmentations for training :

```python
TRAIN_AUG = A.Compose([
    A.RandomResizedCrop(height=IMG_SZ, width=IMG_SZ, p=1.0, scale=(0.72, 1.0)),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=360, border_mode=0, p=0.75),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    A.Blur(p=0.5),
    A.CoarseDropout(max_height=int(32*(IMG_SZ/512)), max_width=int(32*(IMG_SZ/512)), p=0.75),
    A.Normalize(),
    ToTensorV2(),
])
```

Augmentations for validation :

```python
VALID_AUG = A.Compose([
    A.SmallestMaxSize(max_size=IMG_SZ + 16, p=1.0),
    A.CenterCrop(height=IMG_SZ, width=IMG_SZ, p=1.0),
    A.Normalize(),
    ToTensorV2(),
])
```

All the models were trained with Mixup ([implmentation](https://github.com/rwightman/pytorch-image-models/blob/d4ea5c7d7d55967a8bedbfbb58962131d8aba776/timm/data/mixup.py#L90)) with the following settings

```yaml
mixup_alpha: 0.4
cutmix_alpha: 1.
prob: 0.6
switch_prob: 0.5
label_smoothing: 0.00
```
