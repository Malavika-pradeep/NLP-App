import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocessor_text(text):
    #tokenize text
    tokens= word_tokenize(text.lower())


#download nltk corpus
nltk.download('all')

# Load the amazon review dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')


print('AMAZON REVIEW DATASET: ')
print(df.head())




