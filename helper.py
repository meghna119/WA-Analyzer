from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import preprocessor
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set(style="whitegrid")


nltk.download('vader_lexicon')


extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user, df):
    extractor = URLExtract()

    # Read stop words from a file
    with open('stop_words.txt', 'r') as f:
        stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        words = message.lower().split()
        for word in words:
            if word not in stop_words and not extractor.has_urls(word):
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    wordcloud = wc.generate(temp['message'].str.cat(sep=" "))
    return wordcloud


def most_common_words(selected_user,df):

    f = open('stop_words.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.emoji_count(c) > 0])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()





def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    # 1. Check for Empty DataFrames
    if user_heatmap.empty:
        print("user_heatmap is empty")
        return None  # You can return None or handle it based on your app's logic

    # 2. Print Debug Information
    print("Original user_heatmap:")
    print(user_heatmap)
    print("Original user_heatmap shape:", user_heatmap.shape)
    print("Data types before conversion:")
    print("Index:", user_heatmap.index.dtype)
    print("Columns:", user_heatmap.columns.dtype)

    # 3. Ensure Numeric Data
    user_heatmap = user_heatmap.astype(float)

    # 4. Print Debug Information after conversion
    print("user_heatmap after conversion:")
    print(user_heatmap)
    print("user_heatmap shape after conversion:", user_heatmap.shape)
    print("Data types after conversion:")
    print("Index:", user_heatmap.index.dtype)
    print("Columns:", user_heatmap.columns.dtype)

    # 5. Proceed with creating the heatmap
    try:
        ax = sns.heatmap(user_heatmap)
    except ValueError as e:
        print(f"Error: {e}")
        return None  # You can return None or handle it based on your app's logic

    return ax



def analyze_sentiment(df, classifier):
    opinions = {'positive': 0, 'negative': 0, 'neutral': 0}
    for index, row in df.iterrows():
        chat_message = row['message']
        sentiment = preprocessor.classify_sentiment(classifier, chat_message)
        if sentiment == 'positive':
            opinions['positive'] += 1
        elif sentiment == 'negative':
            opinions['negative'] += 1
        else:
            opinions['neutral'] += 1
    return opinions

def sentiment_analysis(df):
    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = []
    for message in df['message']:
        sentiment = analyzer.polarity_scores(message)
        sentiment_scores.append(sentiment)

    df['sentiment'] = sentiment_scores

    return df

def overall_sentiment_analysis(messages):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

    for message in messages:
        sentiment = analyzer.polarity_scores(message)
        if sentiment['compound'] >= 0.05:
            sentiment_counts['positive'] += 1
        elif sentiment['compound'] <= -0.05:
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1

    total_messages = len(messages)
    sentiment_percentages = {k: v/total_messages*100 for k, v in sentiment_counts.items()}

    return sentiment_counts, sentiment_percentages

def monthly_timeline_with_sentiment(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()

    dominant_sentiments = []
    for year, month in zip(timeline['year'], timeline['month']):
        month_df = df[(df['year'] == year) & (df['month'] == month)]
        sentiment_month = overall_sentiment_analysis(month_df['message'])
        dominant_sentiment = max(sentiment_month[0], key=sentiment_month[0].get)
        dominant_sentiments.append(dominant_sentiment)

    timeline['dominant_sentiment'] = dominant_sentiments

    timeline['year'] = timeline['year'].astype(str).str.replace(',', '')

    return timeline

















