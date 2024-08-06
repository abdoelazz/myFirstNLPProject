import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
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

def training(X_train,y_train,cv):
    model1 = GaussianNB()
    model2 = MultinomialNB()
    model3 = BernoulliNB()
    model1.fit(X_train,y_train)
    model2.fit(X_train,y_train)
    model3.fit(X_train,y_train)
    models = model1 , model2 , model3 
    joblib.dump(cv, 'vectorizer.pkl')
    joblib.dump(models, 'models.pkl')
    return model1 , model2 , model3 
    

def test_models(X_test,y_test,models):
    y_pred1=models[0].predict(X_test)
    y_pred2=models[1].predict(X_test)
    y_pred3=models[2].predict(X_test)
    print("Gaussian",accuracy_score(y_test,y_pred1))
    print("Multinomial",accuracy_score(y_test,y_pred2))
    print("Bernaulli",accuracy_score(y_test,y_pred3))

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



    
    


path='./twitter_training.csv'
path2='./twitter_validation.csv'
df=readData(path)
df2=readData(path2)

dataProcessed=df
dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(dataPreprocessing)
dataProcessed['sentiment'] = dataProcessed['sentiment'].apply(lowercase)
dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(remove_special)
dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(remove_stopwords)
dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(stemming)
dataProcessed['Tweet content'] = dataProcessed['Tweet content'].apply(join_back)
cv , X=vectorizer(dataProcessed,'Tweet content')
y=dataProcessed.iloc[:,2].values
models = training(X,y,cv)


dataProcessed2=df2
dataProcessed2['Tweet content'] = dataProcessed2['Tweet content'].apply(dataPreprocessing)
dataProcessed2['sentiment'] = dataProcessed2['sentiment'].apply(lowercase)
dataProcessed2['Tweet content'] = dataProcessed2['Tweet content'].apply(remove_special)
dataProcessed2['Tweet content'] = dataProcessed2['Tweet content'].apply(remove_stopwords)
dataProcessed2['Tweet content'] = dataProcessed2['Tweet content'].apply(stemming)
dataProcessed2['Tweet content'] = dataProcessed2['Tweet content'].apply(join_back)
X = cv.transform(dataProcessed2['Tweet content']).toarray()
y=dataProcessed2.iloc[:,2].values
test_models(X,y,models)



#le meilleur c'est Bernaulli
tweet = "I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣"
dataFrameTest="Tweet content\n"+tweet
model_path = 'models.pkl'
vectorizer_path = 'vectorizer.pkl'
print(predict(dataFrameTest, model_path, vectorizer_path))


