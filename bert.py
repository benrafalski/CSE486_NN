from transformers import pipeline
import sys
import pandas as pd
from datasets import load_dataset, list_datasets

# imdb = load_dataset("imdb")
# print(imdb['train'][0])
# sys.exit(42)

# imdb = load_dataset("imdb")
# sentiment_pipeline = pipeline("sentiment-analysis")
model="finiteautomata/bertweet-base-sentiment-analysis" # 0.7974137931034483
# model = "cardiffnlp/twitter-roberta-base-sentiment" # 0.7672413793103449
# model = "bert-base-uncased" # 0.7629310344827587
sentiment_pipeline = pipeline(model=model)
df = pd.read_csv('data/Comments.csv')
df = df[["Comment", "Sentiment"]]
print(df.head())
def low(text):
    return text.lower()

df['Sentiment'] = df['Sentiment'].apply(low)
print(df.head())

features = df['Comment'].values
labels = df['Sentiment'].values

data = list(zip(features, labels))

print(data[0])

feat, label = data[0]


def label_encoder(label):
    if label == "pos":
        return "label_2" 
    elif label == "neg":
        return "label_0"
    else:
        return "label_1"


total = 0
correct = 0
for f, l in data:
    output = sentiment_pipeline(f)
    # print(f"Expected {l} and got {output[0]['label']}")
    # expected = label_encoder(l.lower()[0:3])
    expected = l.lower()[0:3]
    if expected == 'neu':
        continue
    # print(output)
    # expected = l.lower()
    pred = output[0]['label'].lower()
    if expected == pred:
        correct += 1
    else:
        print(f'Sentence: {f}')
        print(f"\tExpected {expected} and got {pred}\n")
    total += 1

print(correct/total)