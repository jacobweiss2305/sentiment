import pandas as pd
import numpy as np
import tweepy
import re

import gensim
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import JsCode
import streamlit.components.v1 as st_components

bearer_token = ""
auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth, wait_on_rate_limit=True)
sid_obj = SentimentIntensityAnalyzer()

def sentiment_vader(sentence):
    # We don't need to clean text due to this post:
    # https://datascience.stackexchange.com/questions/108094/should-we-clean-text-data-before-applying-vader-for-getting-sentiment

    sentiment_dict = sid_obj.polarity_scores(sentence)
    overall_sentiment = None
    
    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return overall_sentiment

def search_twitter(query, max_tweets, language, twitter_type):
    # Search for tweets
    searched_tweets = []
    last_id = -1
    while len(searched_tweets) < max_tweets:
        count = max_tweets - len(searched_tweets)
        try:
            new_tweets = api.search_tweets(q=query, 
                                           count=count, 
                                           max_id=str(last_id - 1), 
                                           lang=language, 
                                           result_type=twitter_type)
            if not new_tweets:
                break
            searched_tweets.extend(new_tweets)
            last_id = new_tweets[-1].id
        except tweepy.errors.TweepError as e:
            break
    if searched_tweets:
        return searched_tweets

def create_sentiment_df(tweets):
    analysis = pd.DataFrame()
    analysis['Curator'] = [i._json['user']['screen_name'] for i in tweets]
    analysis['Date'] = pd.to_datetime([i._json['created_at'] for i in tweets])
    analysis['Location'] = [i._json['user']['location'] for i in tweets]
    analysis['Tweet'] = [i._json['text'] for i in tweets]
    analysis['Sentiment'] = [i._json['sentiment'] for i in tweets]
    punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
    analysis['Location State'] = np.where((analysis['Location'].str.contains(",") == False), 
                                            np.nan, 
                                            analysis['Location'].apply(lambda x: str(x)[str(x).find(",")+1:].strip().upper()))
    analysis['Location State'] = analysis['Location State'].replace('NAN', np.nan).apply(lambda x: re.sub('[' + punctuation + ']+', ' ', str(x)) if x is not np.nan else x).replace('nan', np.nan)
    return analysis.sort_values(by = 'Date')

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'  # define a string of punctuation symbols

# Functions to clean tweets
def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet

def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def preprocess_tweet(tweet: str, topics: list) -> str:
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""      
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    return tweet

def basic_clean(tweet):
    """Main master function to clean tweets only without tokenization or removal of stopwords"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub('📝 …', '', tweet)
    return tweet

# Streamlit utils
def show_aggrid(df, pagination = True, hyperlink_fields = {}):
    default_length = 15
    page_length = default_length if len(df) > default_length else len(df)

    gb = GridOptionsBuilder.from_dataframe(df)
    if pagination:
        gb.configure_pagination(paginationPageSize=default_length)
    for field in hyperlink_fields.keys():
        gb.configure_column(field,
                            headerName=field,
                            cellRenderer=JsCode('''function(params) {return '<a href='+params.value+' target = "_blank">''' + hyperlink_fields[field] + '''</a>'}'''))
    AgGrid(df, gridOptions=gb.build(), height = 80 + (page_length)*30, enable_enterprise_modules=True,  allow_unsafe_jscode = True)

def download_csv_button(df, list_name):
    name = list_name.replace(' ', '_')
    csv = df.to_csv().encode('utf-8')
    st.download_button(
       "Download CSV",
       csv,
       f"{name}.csv",
       "text/csv",
       key = f'download-csv-{name}'
    )    