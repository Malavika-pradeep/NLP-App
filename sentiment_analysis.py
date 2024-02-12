import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    #tokenize text
    tokens= word_tokenize(text.lower())
     # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the to-kens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
 
    # Join the tokens back into a string
    processed_text= ' '.join(lemmatized_tokens)

    return processed_text




#download nltk corpus
nltk.download('all')

# Load the amazon review dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')


print('AMAZON REVIEW DATASET: ')
print(df.head())


#Preprocess the review text
df['reviewText']= df['reviewText'].apply(preprocess_text)

print(df)

#initialize nlt
analyzer= SentimentIntensityAnalyzer()




