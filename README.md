# HydroViet_VOR
Visual Object Retrieval in satellite images with Triplet Network generated from Template (Processing)


## Data Structure
- Train data and validation data contain 5 classes: dams, buildings, house, river, bridge. Collect by hands from Google Earth (5000 images for now)
- Test data contains 100 images from 5 classes (for now)

```
HydroViet_VOR
|
└───train_data
│   └───house
│       └───house1.jpg
│       └───house2.jpg
│       └───...
│   └───river
│       └───...
|   └───...
|
└───val_data
│   └───house
│       └───house1.jpg
│       └───house2.jpg
│       └───...
│   └───river
│       └───...
|   └───...
|
└───test
    │   1.jpg
    │   2.jpg
    |   ...
```

## Method
- Using Pairwise Ranking Loss model
- Find the distance between 2 feature vector in embedding space. 
- Loss function is MarginRankingLoss (use Euclidean distance to calculate loss)
- Image below is borrowed from https://gombru.github.io/2019/04/03/ranking_loss/


![model](https://i.imgur.com/pXZ2Shi.png)

## Train
- Config hyperparameters in configs/config.py

```
python train.py
```

## Test
- Visualize embeddings (test_data) using tSNE (clustering)

```
python test.py
```

## Visualize
- Using cosine similarity to plot most similar images base on the query image

```
python visualize.py
```

## Result
The best accuracy when validating is 56%, which is not good enough


## Furthur Improvement
Using GANs to train CNN models like EfficientNet, Resnet on satellite data


## Sources
- https://github.com/vltanh/pytorch-reidentification
