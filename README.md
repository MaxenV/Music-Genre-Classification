# Music-Genre-Classification

A model that classifies music styles based on recordings. <br>

Created to learn how to learn models using various techniques and preprocessing audio data.


# How to start
Create new environment: conda env create -f environment.yml

## Create folder tree
- data
    - audio
    - models
    - processed

Move data from kaggle folder "genres_original" into Audio folder
Remove jazz.00054.wav because is damaged.

## Create preprocessed data
Run preproces.ipynb notebook

This will preprocessed data from Audio and create .json file in Processed folder

## Create model
Run model notebook

This will create and learn model using data from Processed folder and save the model in Models

## Dataset 

## Classes:
- blues  
- classical  
- country  
- disco  
- hiphop  
- jazz  
- metal  
- pop  
- reggae  
- rock

## Data source:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification


