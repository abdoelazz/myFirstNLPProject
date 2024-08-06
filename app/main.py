import re
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from io import StringIO
import joblib


def readData(path):
    df=pd.read_csv(path)
    return df

#supprimer les RT de retweet et supprimer les hyperlinks
def dataPreprocessing(text):
    clean=str(text)
    clean=clean.lower()
    clean = re.sub(r'^RT[\s]+', '', clean)
    clean = re.sub(r'https?://[^\s\n\r]+', '', clean)
    return clean

#rendre tout les mots en minuscule
def lowercase(text):
    text_miniscule=text.lower()
    return text_miniscule

#supprimer les symboles qui ne sont ni des alphabets ni des nombres 
def remove_special(text):
    x=''
    
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x + ' '
    return x


#supprimer les stopwords
def remove_stopwords(text):
    x=[]
    
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y

#normaliser les mots
def stemming(text):
    stemmer = PorterStemmer()
    tweets_stem = [] 

    for word in text:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
    return tweets_stem
#rassembler la liste
def join_back(list_input):
    return " ".join(list_input)

def vectorizer(df,column):
    cv=CountVectorizer(max_features=2500)
    X=cv.fit_transform(df[column]).toarray()
    return cv , X


    
def predict(tweet, model_path, vectorizer_path):
    # Load the vectorizer and model
    cv = joblib.load(vectorizer_path)
    models = joblib.load(model_path)
    
    # Create a DataFrame from the input tweet
    dataProcessed = pd.read_csv(StringIO(tweet))
    
    # Apply preprocessing steps
    dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(dataPreprocessing)
    dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(remove_special)
    dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(remove_stopwords)
    dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(stemming)
    dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(join_back)
    
    # Vectorize the processed text using the loaded vectorizer
    X = cv.transform(dataProcessed['Tweet content']).toarray()
    
    # Make prediction using the desired model
    model = models[1] 
    prediction = model.predict(X)
    
    return prediction

app = FastAPI()

class TweetRequest(BaseModel):
    tweet: str

@app.get("/")
def health_check():
    return {"health_check": "OK"}


@app.post("/predict")
def predict_tweet(request: TweetRequest):
    tweet = request.tweet
    dataFrameTest = f"Tweet content\n{tweet}"
    model_path = 'models.pkl'  # Update the path if necessary
    vectorizer_path = 'vectorizer.pkl'  # Update the path if necessary
    
    # Get the prediction
    prediction = predict(dataFrameTest, model_path, vectorizer_path)
    
    predicted_label = prediction[0]

    return {"prediction": predicted_label}
    
