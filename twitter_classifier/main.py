import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tweepy

if not nltk.download('punkt', quiet=True) or not nltk.download('stopwords', quiet=True) or not nltk.download('wordnet', quiet=True):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def get_tweets(api_key, api_secret_key, access_token, access_token_secret, query, count=100):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = []
    for tweet in tweepy.Cursor(api.search, q=query, count=count, lang="en").items():
        tweets.append(tweet.text)
    
    return tweets

def preprocess_tweets(tweets):
    preprocessed_tweets = []

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for tweet in tweets:
        cleaned_tweet = re.sub(r'http\S+|www\S+|https\S+|\[.*?\]|\W|\d+', ' ', tweet)
        words = word_tokenize(cleaned_tweet.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_tweet = ' '.join(filtered_words)
        preprocessed_tweets.append(preprocessed_tweet)

    return preprocessed_tweets

def train_text_classifier(features, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    return classifier, vectorizer, X_test, y_test

def evaluate_model(classifier, X_test, y_test):
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

def predict_categories(new_tweets, vectorizer, classifier):
    preprocessed_tweets = preprocess_tweets(new_tweets)
    X_new = vectorizer.transform(preprocessed_tweets)
    predictions = classifier.predict(X_new)
    return predictions

if __name__ == "__main__":
    api_key = os.environ.get('TWITTER_API_KEY')
    api_secret_key = os.environ.get('TWITTER_API_SECRET_KEY')
    access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
    
    query = '#MachineLearning'

    tweets = get_tweets(api_key, api_secret_key, access_token, access_token_secret, query)

    preprocessed_tweets = preprocess_tweets(tweets)

    labels = [0, 1, 0, 1, ...]  # Example labels (0 and 1)
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_tweets, labels, test_size=0.2, random_state=42)

    classifier, vectorizer, X_test, y_test = train_text_classifier(X_train, y_train)

    evaluate_model(classifier, X_test, y_test)

    new_tweets = ['New tweet 1', 'New tweet 2', ...]
    predictions = predict_categories(new_tweets, vectorizer, classifier)

    for tweet, category in zip(new_tweets, predictions):
        print(f'Tweet: {tweet}\nCategory: {category}\n')
