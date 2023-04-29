#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import pipeline
import sys
import pandas as pd
from datasets import load_dataset, list_datasets

# get the pipeline model from transformers
def get_pipeline(model):
    return pipeline(model=model)
# convert text to lowercase
def low(text):
    return text.lower()
# reads the data from the spreadsheet we made
def get_data():
    df = pd.read_csv('data/Comments.csv')
    df = df[["Comment", "Sentiment"]]
    df['Sentiment'] = df['Sentiment'].apply(low)
    features = df['Comment'].values
    labels = df['Sentiment'].values
    data = list(zip(features, labels))
    return data
# encodes the labels for the model "finiteautomata/bertweet-base-sentiment-analysis" 
def label_encoder(label):
    if label == "pos":
        return "label_2" 
    elif label == "neg":
        return "label_0"
    else:
        return "label_1"
# test the accruacy of the pipeline using the data provided
def test_pipiline(sentiment_pipeline, data):
    total = 0
    correct = 0
    for f, l in data:
        output = sentiment_pipeline(f)
        expected = l.lower()[0:3]
        pred = output[0]['label'].lower()
        if expected == pred:
            correct += 1
        total += 1
    return correct/total

def main():
    model="finiteautomata/bertweet-base-sentiment-analysis" # 0.7974137931034483
    pipeline = get_pipeline(model=model)
    data = get_data()
    print("Starting testing...")
    accuracy = test_pipiline(sentiment_pipeline=pipeline, data=data)
    print(f'Model accracy is {accuracy}')

if __name__ == "__main__":
    main()
