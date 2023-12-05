import re
import pandas as pd
import regex

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df


    #
    def extract_emojis(text):
        emoji_pattern = regex.compile("["
                                      u"\U0001F600-\U0001F64F"  # emoticons
                                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                      "]+", flags=regex.UNICODE)
        emojis = ''.join(emoji_pattern.findall(text))
        return emojis
    #
    # def map_emoji_sentiments(emojis):
    #     positive_emojis = ['😊', '😄', '😃', '🤣', '😂']  # Define positive emojis
    #     negative_emojis = ['😢', '😞', '😔', '😭']  # Define negative emojis
    #     neutral_emojis = ['😐', '😑', '😶']  # Define neutral emojis
    #
    #     # Check sentiment for each emoji individually
    #     positive_count = sum(1 for emoji in emojis if emoji in positive_emojis)
    #     negative_count = sum(1 for emoji in emojis if emoji in negative_emojis)
    #     neutral_count = sum(1 for emoji in emojis if emoji in neutral_emojis)
    #
    #     # Determine sentiment based on counts
    #     if positive_count > negative_count and positive_count > neutral_count:
    #         return 'Positive'
    #     elif negative_count > positive_count and negative_count > neutral_count:
    #         return 'Negative'
    #     else:
    #         return 'Neutral'
    #
    # # def map_emoji_sentiments(emojis):
    # #     happy_emojis = ['😊', '😄', '😃', '🤣', '😂']  # Define your happy emojis
    # #     sad_emojis = ['😢', '😞', '😔', '😭']  # Define your sad emojis
    # #     neutral_emojis = ['😐','😑','😶']
    # #
    # #
    # #     # Your logic to map emojis to sentiments (e.g., happy, sad, neutral)
    # #     if emojis in happy_emojis:
    # #         return 'Happy'
    # #     elif emojis in sad_emojis:
    # #         return 'Sad'
    # #     elif emojis in neutral_emojis:
    # #         return 'Neutral'
    # #     else:
    # #         return 'Undetermined'
    # #
    # #
    # df['emojis'] = df['message'].apply(extract_emojis)  # Extract emojis
    # df['emoji_sentiments'] = df['emojis'].apply(map_emoji_sentiments)
    #


    sentiment_scores = []
    for message in df['message']:
        analysis = TextBlob(message)
        sentiment_scores.append(analysis.sentiment.polarity)  # Extract sentiment score

    df['SentimentScore'] = sentiment_scores  # Add sentiment scores as a new column

    # Define thresholds
    negative_threshold = -0.5
    positive_threshold = 0.5

    # Convert SentimentScore to categorical labels
    df['SentimentClass'] = pd.cut(df['SentimentScore'],
                                  bins=[float('-inf'), negative_threshold, positive_threshold, float('inf')],
                                  labels=['Negative', 'Neutral', 'Positive'])

    return df

