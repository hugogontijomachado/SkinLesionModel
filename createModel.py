import torch
import torch.nn as nn
from fastai import *
from fastai.vision import *
from fastai.vision.all import * 
from fastai.metrics import *
#from fastai.vision.aug_transforms import *
#import torchvision.transforms.functional as TTF
import os



def modelGen(dfs, TRAIN_PATH, bs=128):
    """
    Esta função criado os objetos "Dataloader" e "Lerner" 
    que iremos utilizar para criar nosso modelo preditivo
    dfs :: lista contendo os datasets divididos
    TRAIN_PATH :: diretório local contendo as imagens do ISIC 2020
    bs :: "batch size" do dataloader
    """
    
    
    ## Funções get_x e get_y
    def get_x(df):
        image_name = df['image_name']
        return os.path.join(TRAIN_PATH,f'{image_name}.jpg')       
    def get_y(df):
        return df['target']
    
    ## Datablock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(224),
        batch_tfms=[
            *aug_transforms(do_flip=0.5, flip_vert=0.5),
            Normalize.from_stats(*imagenet_stats)
        ]
    )

    ## Dataloaders
    dls = []
    for df in dfs:
        dls.append(
            dblock.dataloaders(df,bs=bs)
        )

    ## Learners
    learners = []
    for dl in dls:
        learners.append(
            cnn_learner(
                dl,
                resnet18,
                metrics=[
                    accuracy,
                    #RocAucBinary(average='weighted'),
                    #Precision(average='weighted'),
                    #F1Score(average='weighted'),
                    Recall(average='weighted')
                ]
            )
        )
    return dls, learners 
    