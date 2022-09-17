from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import seaborn as sns

import utils

FIG_SIZE=(6, 4)
PALETTE = {
    'Positive': '#0FF0FC',
    'Neutral': '#808080',
    'Negative': '#FF3131',
}
plt.style.use("dark_background")

st.set_page_config(page_title='Twitter Sentiment', layout='wide')
                   
st.header("Twitter Sentiment Analysis")

# Set sidebar options
st.sidebar.header("Search parameters")
topic = st.sidebar.text_input("Search topic", "joe rogan")
max_tweets = st.sidebar.slider("Max tweets", 1_000, 10_000, step=1_000, value = 1_000)
result_type_options = {"Popular and Real-time":"mixed", "Real-time":"recent",  "Popular":"popular"}
twitter_type = st.sidebar.selectbox("Tweet type", result_type_options.keys())
languages = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it', 'Dutch': 'nl', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko', 'Chinese': 'zh'}
language_type = st.sidebar.selectbox("Language", languages.keys())
filter_retweets = st.sidebar.selectbox("Filter retweets", [True, False])
filter_replies = st.sidebar.selectbox("Filter replies", [True, False])
start_date = st.sidebar.date_input("Start date", datetime.today())
days_ago = st.sidebar.slider("Days ago", 1, 30, value=15)
end_date = st.sidebar.date_input("End date", (datetime.today() - timedelta(days=days_ago)))

# set query
base = []
if topic:
    base.append(topic)
if filter_retweets:
    base.append('-filter:retweets')
if filter_replies:
    base.append('-is:reply')
if start_date:
    base.append(f'until:{start_date}')    
if end_date:
    base.append(f'since:{end_date}')
if len(base) > 0:    
    query = ' '.join(base)
st.sidebar.text_input("Twitter API Query (debugging)", query)

# Search twitter
tweets = utils.search_twitter(query, 
                              int(max_tweets), 
                              languages[language_type], 
                              result_type_options[twitter_type])
if tweets:
    st.text(f"Total tweets: {len(tweets)}")
    for tweet in tweets:
        tweet._json["sentiment"] = utils.sentiment_vader(tweet._json["text"])
    analysis = utils.create_sentiment_df(tweets)
    
###################
#      Row 1
###################

col1, col2, col3 = st.columns(3)
with col1:
    # Total sentiment count
    fig = plt.figure()
    sentiment_count = analysis.groupby('Sentiment').Tweet.count().reset_index().sort_values(by='Sentiment', ascending=False)
    sns.barplot(data=sentiment_count, x="Sentiment", y="Tweet", palette=PALETTE)
    plt.title(f"Total Sentiment Count")
    st.pyplot(fig)

with col2: 
    # Sentiment Trend
    COLORS = ["#0FF0FC", "#808080", "#FF3131"] # positive, neutral, negative
    analysis['Date'] = pd.to_datetime(analysis['Date']).dt.strftime('%Y-%m-%d')
    trend = analysis.groupby('Date').Sentiment.value_counts(normalize=True).unstack()
    fig = plt.figure(figsize=FIG_SIZE)
    if 'Positive' in trend.columns:
        sns.lineplot(y=trend['Positive'], x=trend.index, color = "#0FF0FC")
    if 'Negative' in trend.columns:        
        sns.lineplot(y=trend['Negative'], x=trend.index, color = "#FF3131")
    if 'Neutral' in trend.columns:
        sns.lineplot(y=trend['Neutral'], x=trend.index, color = "#808080")
    plt.ylabel("% of Tweets")
    plt.title(f"Sentiment Trend")
    a = Line2D([], [], color="#0FF0FC", label="Positive")
    b = Line2D([], [], color="#FF3131", label="Negative")
    c = Line2D([], [], color="#808080", label="Neutral")
    plt.legend(handles=[a, b, c], title = "Sentiment")            
    plt.xticks(rotation=30)
    st.pyplot(fig)

with col3:
    # Sentiment by location
    sentiment_by_location = analysis[['Location State', 'Sentiment']].value_counts().head(10).reset_index()
    sentiment_by_location.columns = ['Location State', 'Sentiment', 'Count']
    fig = plt.figure()
    sns.barplot(x=sentiment_by_location['Location State'], y=sentiment_by_location['Count'], hue=sentiment_by_location['Sentiment'], palette=PALETTE)    
    plt.title("Sentiment by location")
    plt.legend(loc='upper right')
    st.pyplot(fig)

with col1:
    st.text("")
with col2:
    st.text("")
with col3:
    st.text("")        
    
###################
#      Row 2
###################

with col1:
    # Popular topics
    if len(topic) > 1:
        topics = topic.split()
    cleaned_tweets = list(analysis['Tweet'].apply(lambda x: utils.preprocess_tweet(x, topics)))
    vectorizer = TfidfVectorizer(stop_words='english')
    tf = vectorizer.fit_transform(cleaned_tweets)

    tfidf_feature_names = vectorizer.get_feature_names()
    nmf = NMF(n_components=3, random_state=1).fit(tf)

    n_top_words = 3
    topics = []
    popularity = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        topics.append(top_features)
        popularity.append(weights)
        
    topics_df = pd.DataFrame([i for j in topics for i in j], columns=['Topic'])
    values_df = pd.DataFrame([i for j in popularity for i in j], columns=['Popularity Score'])
    results = pd.concat([topics_df, values_df], axis=1).head(5)
    results.sort_values(by='Popularity Score', ascending = False, inplace=True)

    fig = plt.figure()
    sns.barplot(x=results["Popularity Score"], y=results["Topic"], orient='h', palette="pastel")
    plt.xlabel('Popularity Score')
    plt.ylabel('Trending Topics')
    plt.title('Popular Topics')
    st.pyplot(fig)

with col2:
    # Sentiment by topic
    topics = list(results["Topic"])[:5]
    collect = []
    for i in topics:
        analysis[f"{i}"] = analysis["Tweet"].str.contains(i, case=False).astype(int)
        temp = analysis[analysis[f"{i}"] == 1]
        sentiment_count = (temp.groupby('Sentiment').Tweet.count()/temp.shape[0]).reset_index().sort_values(by='Sentiment', ascending=False)
        sentiment_count.index = [i] * sentiment_count.shape[0]
        collect.append(sentiment_count)
    df = pd.concat(collect).reset_index()
    df.columns = ['Topic', 'Sentiment', 'Percentage']

    fig = plt.figure()
    sns.barplot(x='Percentage', y='Topic', data=df, hue='Sentiment', orient='h', palette=COLORS)
    plt.title("Sentiment by Topic")
    plt.legend(loc='upper right')
    st.pyplot(fig)

with col3:
    # Top Topic by location
    topics = list(results["Topic"])[:5]
    collect = []
    for i in topics:
        analysis[f"{i}"] = analysis["Tweet"].str.contains(i, case=False).astype(int)
        temp = analysis[analysis[f"{i}"] == 1]
        sentiment_count = (temp.groupby('Location State').Tweet.count()/temp.shape[0]).reset_index().sort_values(by='Location State', ascending=False)
        sentiment_count.index = [i] * sentiment_count.shape[0]
        collect.append(sentiment_count)
    df = pd.concat(collect).reset_index()
    df.columns = ['Topic', 'Location State', 'Percentage']
    top_topic_by_state = df.sort_values(by = ['Percentage']).groupby('Topic').tail(1)
    fig = plt.figure()
    sns.barplot(x='Percentage', y='Topic', data=top_topic_by_state, hue='Location State', orient='h',)
    plt.title("Top Topic by location")
    st.pyplot(fig)
    
with col1:
    st.text("")
with col2:
    st.text("")
with col3:
    st.text("")
    
###################
#      Row 3
###################

with col1:
    # Total sentiment count
    fig = plt.figure()
    curator_count = analysis.groupby('Curator').Tweet.count().reset_index().sort_values(by='Tweet', ascending=False)
    sns.barplot(data=curator_count.head(5), x="Tweet", y="Curator", orient='h')
    plt.title(f"Top Curators")
    st.pyplot(fig)

with col2:
    # Sentiment by Curator
    top_5_curators = list(curator_count.Curator.head(5))
    sentiment_by_curator = analysis[analysis['Curator'].isin(top_5_curators)][['Curator', 'Sentiment']].value_counts(normalize = True).reset_index()
    sentiment_by_curator.columns = ['Curator', 'Sentiment', 'Percentage']
    fig = plt.figure()
    sns.barplot(x=sentiment_by_curator['Percentage'], y=sentiment_by_curator['Curator'], hue=sentiment_by_curator['Sentiment'], palette=PALETTE)    
    plt.title("Curator Sentiment")
    plt.legend(loc='upper right')
    st.pyplot(fig) 


if not analysis.empty:
    # Download data
    st.subheader("Download data")
    utils.download_csv_button(df, "tweets")
    columns = ['Curator', 'Date', 'Tweet', 'Location', 'Sentiment']
    utils.show_aggrid(analysis[columns])