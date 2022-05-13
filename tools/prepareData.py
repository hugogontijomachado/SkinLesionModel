import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import StratifiedKFold



def removeDuplicates(df,df_dup):
    """
    Esta função irá excluir as imagens repetidas do dataframe df 
    df :: dataframe de todas as imagens de treino ISIC 2020
    df_dup :: datagrame com as imagens duplicadas do dataset
    return :: retorna o dataset com as linhas que contém imagens duplicadas devidamente excluidas
    """
    
    for im in list(df_dup['image_name_2']):
        df.drop( df.index[df['image_name'] == im], inplace = True )
        df.reset_index(drop=True, inplace=True)
    return df

def splitDataset(df, nSplits=5, frac=0.15):
    """
    Devido ao grande desbalanceamento entre malignos e benignos, esta função divide os casos
    benignos, retira uma fração deles, e concatena cada um com os casos malignos

    nSplits :: número de divisões dos casos benignos
    frac :: fração dos casos benignos para cada dataset dividido
    return :: retorna uma lista com os datasets construidos
    """
    ## Separando malignos de benignos
    dfMalign = df[df['target']==1]
    dfMalign.reset_index(drop=True,inplace=True)
    dfBenign = df[df['target']==0]
    dfBenign.reset_index(drop=True,inplace=True)

    ## Criando a coluna "fold" que irá conter de qual dataset benigno divido pertence cada linha
    skf = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)
    for fold, (t_, v_) in enumerate(skf.split(X=dfBenign.values, y=dfBenign.target.values)):
        dfBenign.loc[v_, 'fold'] = fold

    ## Concatenando cada dataset benigno com o dataset maligno
    dfs = []
    for i in range(5):
        dfs.append(
            pd.concat([
                dfBenign[dfBenign['fold'] == i].sample(frac=0.15,random_state=42),
                dfMalign
            ])
        )
    dfs[i] = dfs[i].sample(frac=1)
    dfs[i] = dfs[i].reset_index(drop=True)
    dfs[i]['fold'] = i

    return dfs



