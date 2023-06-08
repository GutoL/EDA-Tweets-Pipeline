from google.cloud import bigquery
from google.oauth2 import service_account
import codecs
import csv
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import regex
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import liwc
from scipy.signal import find_peaks
import os
from utils.peak_detection import PeakDetector
import numpy as np

plt.style.use('ggplot')

def save_df(query_job, result_file_name):

    result = query_job.result()

    schema = result.schema

    file_encoding="utf-8-sig"

    with codecs.open(result_file_name, "w", encoding=file_encoding) as f:
        writer = csv.writer(f)
        # Write headers
        header = [f_name.name for f_name in schema ]
        writer.writerow(header)
        # Write data to file
        for row in query_job:
            writer.writerow(row)

def save_df_chat(query_job, output_file_path):
    result = query_job.result()

    with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([field.name for field in result.schema])
        for row in result:
            writer.writerow(row)


class EdaTextAnalysis():
    def __init__(self, database_link, credentials, project_id, results_path):
        self.database_link = database_link
        self.credentials = credentials
        self.project_id = project_id
        self.results_path = results_path

    def run_query(self, query, result_file_name=False):

        client = bigquery.Client(credentials=self.credentials, project=self.project_id)
        
        # print('Running the query...')
        query_job = client.query(query)#.to_dataframe()
        
        # for row in query_job:
        #     print(type(row))
        #     break

        if result_file_name:
            # print('Done!\nSaving...')
            save_df_chat(query_job, result_file_name)
        else:
            data = query_job.to_dataframe()
            return data

    def create_query(self, fields_to_be_recovered, where_clause):
        query = 'SELECT '  

        for x, field in enumerate(fields_to_be_recovered):
            
            if x < len(fields_to_be_recovered)-1:
                query += fields_to_be_recovered[field]+', '
            else:
                query += fields_to_be_recovered[field]+' '
        
        query += 'FROM `'+self.database_link+ '` '+where_clause

        return query    

    def number_of_tweets_query(self):
        query = "SELECT COUNT(*) FROM `"+self.database_link+"`"
        return self.run_query(query, result_file_name=False).iloc[0][0]

    def number_of_users_query(self, author_column, where_clause=''):
        query = "SELECT COUNT(DISTINCT("+author_column+")) FROM `"+self.database_link+"`"+where_clause
        return self.run_query(query, result_file_name=False).iloc[0][0]

    def number_of_original_tweets_query(self, text_column):
        query = "SELECT COUNT("+text_column+") FROM `"+self.database_link+"` WHERE ("+text_column+" IS NOT NULL AND "+text_column+' NOT LIKE "RT%" and '+text_column+' NOT LIKE "@%")'
        return self.run_query(query, result_file_name=False).iloc[0][0]


    def retweets_query(self, fields, limit=False):
        tweet_field = fields['text_tweet'] # fields[1]
        reply_clause = 'WHERE '+tweet_field+' IS NOT NULL AND '+tweet_field+' LIKE "RT%"'
        
        if limit:
            reply_clause += ' LIMIT '+str(limit)
    
        return self.run_query(self.create_query(fields, reply_clause), result_file_name=self.results_path+'retweets.csv')
    

    def number_of_retweets_query(self, text_column, count=True):
        query = "SELECT COUNT("+text_column+") FROM `"+self.database_link+"` WHERE ("+text_column+" IS NOT NULL AND "+text_column+' LIKE "RT%")'
        result = self.run_query(query, result_file_name=False)

        if count:
            return result.iloc[0][0]
        else:
            return result

    def replies_query(self, fields, limit=100):
        tweet_field = fields['text_tweet'] # fields[1]
        reply_clause = 'WHERE '+tweet_field+' IS NOT NULL AND '+tweet_field+' LIKE "@%"'+'LIMIT '+str(limit)
        
        return self.run_query(self.create_query(fields, reply_clause), result_file_name=self.results_path+'replies.csv')

    def number_of_replies_query(self, text_column, count=True):
        query = "SELECT COUNT("+text_column+") FROM `"+self.database_link+"` WHERE ("+text_column+" IS NOT NULL AND "+text_column+' LIKE "@%")'
        
        result = self.run_query(query, result_file_name=False)

        if count:
            return result.iloc[0][0]
        else:
            return result     


    def most_active_users_query(self, text_column, user_column, limit=10000):
        query = "SELECT "+user_column+", COUNT("+text_column+") as Number_of_Tweets FROM `"+self.database_link+"` WHERE "+text_column+" IS NOT NULL GROUP BY "+user_column+" ORDER BY Number_of_Tweets DESC LIMIT "+str(limit)

        return self.run_query(query, result_file_name=False)

    def most_visible_users_query(self, fields, limit=10000):
        self.replies_query(fields, limit=limit)
        self.retweets_query(fields, limit=limit)
        ress = subprocess.call("Rscript utils/find_VisibleUsers.R "+self.results_path, shell=True)
    

    def verified_users_query(self,username_column, verified_column, limit=10000):
        query = "SELECT DISTINCT("+username_column+") FROM `"+self.database_link+"` WHERE "+verified_column+' = true' # = '"true"'

        if limit:
            query += ' LIMIT '+str(limit)

        return self.run_query(query=query, result_file_name=False)


    def volume_of_tweets_per_user_type_query(self, username_column, user_type='active', date1=None, date2=None, 
                                verified_column=None, limit=100):
        
        query = 'SELECT * FROM `'+self.database_link+'` WHERE '+username_column+' IN ('

        if user_type == 'active':
            df = pd.read_csv(self.results_path+'most_active_users.csv')

            if limit and df.shape[0] > limit:
                df = df.head(limit)
            
        elif user_type == 'visible':
            df = pd.read_csv(self.results_path+'Visible_Users.csv')
            df.rename(columns={"Target": username_column}, errors="raise", inplace=True)

            if limit and df.shape[0] > limit:
                df = df.head(limit)
        
        else: # verified
            df = self.verified_users_query(username_column, verified_column, limit)

        users_list = list(df[username_column].values)

        for i, user in enumerate(users_list):
            if i == len(users_list)-1:
                query += '"'+user+'")'
            else:
                query += '"'+user+'", '

        
        return self.run_query(query=query, result_file_name=False)


    def tweets_per_country_query(self, country_column, country_code):
        query = "SELECT COUNT(*) FROM `"+self.database_link+"` WHERE "+country_column+"='"+country_code+"'"

        return self.run_query(query, result_file_name=False)

    def users_per_country_query(self, username_column, country_column, country_code):
        query = "SELECT DISTINCT("+username_column+") FROM `"+self.database_link+"` WHERE "+country_column+"='"+country_code+"'"

        return self.run_query(query, result_file_name=False)
    
    def tweets_per_language_query(self, language_column, language_code):
        query = "SELECT COUNT(*) FROM `"+self.database_link+"` WHERE "+language_column+"='"+language_code+"'"

        return self.run_query(query, result_file_name=False).values[0][0]

    def volume_of_hashtags_per_date_query(self, date_column, hashtag_column):
        query = 'SELECT '+date_column+', COUNT(DISTINCT(hashtag)) as num_hashtags FROM `'+self.database_link+'`, UNNEST(SPLIT('+hashtag_column+', ",")) as hashtag WHERE hashtag != "" GROUP BY '+date_column+' ORDER BY '+date_column+' ASC'
        df = self.run_query(query=query, result_file_name=False)

        plt.figure(figsize=(10, 6))

        plt.plot(df[date_column], df['num_hashtags'])
        plt.xlabel('Date')

        plt.ylabel('Number of Hashtags')

        plt.title('Number of Hashtags per Date')
        # plt.show()
        plt.savefig(self.results_path+'hashtags_per_day.png')

        return df

    def volume_of_hashtags_query(self, hashtag_column):
        query = 'SELECT DISTINCT hashtag FROM `'+self.database_link+'`, UNNEST(SPLIT('+hashtag_column+', "," )) as hashtag WHERE hashtag != "" '
        return self.run_query(query=query, result_file_name=False)

    def all_urls_query(self, url_column):
        query = 'SELECT '+url_column+' FROM `'+self.database_link+'` WHERE '+url_column+' IS NOT NULL'
        return self.run_query(query, False)
    
    def number_of_source_query(self, source_column):
        query = 'SELECT DISTINCT('+source_column+') FROM `'+self.database_link+'` '
        return self.run_query(query=query, result_file_name=False)

    def number_of_tweets_per_source_query(self, source_column):
        query = 'SELECT '+source_column+', COUNT(*) as tweet_count FROM `'+self.database_link+'` GROUP BY '+source_column+' ORDER BY tweet_count DESC'
        return self.run_query(query, False)

    def preprocess(self, tweets):
        # define a regex pattern to match urls, hashtags, special characters, punctuations, numbers, and emojis
        url_pattern = regex.compile(r'https?://\S+|www\.\S+')
        hashtag_pattern = regex.compile(r'#\w+')
        special_chars_pattern = regex.compile(r'[^\w\s]')
        punctuations_pattern = regex.compile(r'[^\w\s]|_')
        numbers_pattern = regex.compile(r'\d+')
        emojis_pattern = regex.compile('[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', flags=regex.UNICODE)

        # initialize a list to store the n-grams
        preprocessed_text = []        

        # loop through each tweet
        for tweet in tweets:
            # remove urls, hashtags, special characters, punctuations, numbers, and emojis from the tweet
            tweet = regex.sub(url_pattern, '', tweet)
            tweet = regex.sub(hashtag_pattern, '', tweet)
            tweet = regex.sub(special_chars_pattern, '', tweet)
            tweet = regex.sub(punctuations_pattern, '', tweet)
            tweet = regex.sub(numbers_pattern, '', tweet)
            tweet = regex.sub(emojis_pattern, '', tweet)

            preprocessed_text.append(tweet)

        return preprocessed_text
    
    def generate_ngrams(self, text, n):
        # Tokenize the text into words
        tokens = word_tokenize(text.lower())
        
        # initialize a Porter stemmer
        stemmer = PorterStemmer()
        
        # define stopwords to be removed and remove stopwords and stem the tokens
        stop_words = set(stopwords.words('english'))
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        
        # Generate the ngrams
        ngrams_list = ngrams(tokens, n)
        
        # Join the tokens in each ngram to form a string
        ngrams_strings = [' '.join(grams) for grams in ngrams_list]
        
        return ngrams_strings
    
    def get_top_ngrams(self, texts, n, top_n):
        # Combine all texts into a single string
        all_text = ' '.join(texts)
        
        # Generate the ngrams for all_text
        ngrams_list = self.generate_ngrams(all_text, n)
        
        # Count the frequency of each ngram
        ngrams_count = Counter(ngrams_list)
        
        # Return the top n most common ngrams
        return ngrams_count.most_common(top_n)

    def run_n_gram(self, text_column, n, top_n, limit):

        query = 'SELECT '+text_column+' FROM `'+self.database_link+'`'

        if limit:
            query += ' LIMIT '+str(limit)

        df = self.run_query(query, False)

        preprocessed_text = self.preprocess(tweets=df[text_column].values)

        ngrams = self.get_top_ngrams(preprocessed_text, n=n, top_n=top_n)

        return ngrams
    
    def peak_detection(self, num_peaks, category=''):
        query = "SELECT created_at FROM `"+self.database_link+"`"
        df = self.run_query(query, result_file_name=False) #.iloc[0][0]
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.set_index('created_at', inplace=True)
        
        tweets = df.resample('D').size()
        
        peak_i = np.argpartition(tweets, -int(num_peaks))[-int(num_peaks):]
        
        peak_values = tweets.values[peak_i]
        peak_dates = tweets.index[peak_i]
        plt.plot(tweets.index, tweets.values)
        
        plt.scatter(peak_dates, peak_values, label='Peaks')
        
        plt.xlabel('Date and Time')
        plt.ylabel('Number of Tweets')
        plt.title(category)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.results_path+f'{category}_plot_pd.pdf')

        # return plt.show()


    def generate_wordcloud(self, ngrams_list):
        # Create a dictionary mapping ngrams to their frequency
        ngrams_dict = dict(ngrams_list)
        
        # Create a WordCloud object with the ngrams_dict
        wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='Blues').generate_from_frequencies(ngrams_dict)
        
        # Display the word cloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(self.results_path+'word_cloud.png')

    def analyse_sentiment(self, tweet_id_column, text_column, limit):
        query = 'SELECT '+tweet_id_column+','+text_column+' FROM `'+self.database_link+'`'

        if limit:
            query += ' LIMIT '+str(limit)

        df = self.run_query(query, False)

        """
        Analyze the sentiment of a list of tweets using TextBlob.
        Returns a list of tuples containing the tweet and its sentiment score.
        """
        sentiments = []
        for tweet in df[text_column].values:
            analysis = TextBlob(tweet)

            sentiment = analysis.sentiment.polarity
            # sentiments.append((tweet, sentiment))
            sentiments.append(sentiment)
        
        df['sentiment'] = sentiments
        # It returns a list of tuples, where each tuple contains the tweet and its sentiment score:
        # Note that TextBlob assigns a polarity score between -1 and 1 to each tweet, where -1 indicates negative sentiment, 
        # 0 indicates neutral sentiment, and 1 indicates positive sentiment.
        return df

    def run_liwc(self, text_column, dict_name, limit):
        query = 'SELECT '+text_column+' FROM `'+self.database_link+'`'+' LIMIT '+str(limit)

        df = self.run_query(query, False)

        parse, category_names = liwc.load_token_parser(dict_name)

        print(category_names)

        df['preprocess'] = self.preprocess(df[text_column].values)

        #Tokenize and drop NaN tokens
        split_data = df['preprocess'].str.split(" ")
        df['tokens'] = split_data
        nan_value = float("NaN")
        df.replace("", nan_value, inplace=True)
        df.dropna(subset = ["tokens"], inplace=True)

        #LIWC Features Extraction
        liwc_results =[] 
        for item in df.tokens:
            print(item)
            # gettysburg_counts = list(collections.Counter(category for token in item for category in parse(token)))
            gettysburg_counts = Counter(category for token in item for category in parse(token))

            liwc_results.append(gettysburg_counts)
        
        print(liwc_results)
        # liwc_ = np.array(liwc_results)
        # df['family'] = liwc_

        # print(df['family'])

# --------------------------------------------------------------------------------------------------

def pipeline(setup_file_name, fields):

    f = open(setup_file_name)
    
    config = json.load(f)

    results_path = config['results_path']
    
    credentials = service_account.Credentials.from_service_account_file(
        config['key_path'],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    eda = EdaTextAnalysis(database_link=config['database_link'], credentials=credentials, project_id=config['project_id'], results_path=results_path)

    results = []

    print('Starting the pipeline...')

    number_of_tweets = int(eda.number_of_tweets_query())
    print('Number of tweets:', number_of_tweets)
    results.append( {'Metric':'Number of tweets', 'Value':number_of_tweets})

    number_of_users = eda.number_of_users_query(author_column=fields['user_username'])
    print('Number of users:', number_of_users)
    results.append({'Metric':'Number of users', 'Value':number_of_users})

    number_of_original_tweets = int(eda.number_of_original_tweets_query(text_column=fields['text_tweet']))
    print('Number of original tweets:', number_of_original_tweets, '➝', (100*number_of_original_tweets)/number_of_tweets, '%')
    results.append({'Metric':'Number of original tweets', 'Value':number_of_original_tweets})

    number_of_retweets = int(eda.number_of_retweets_query(text_column=fields['text_tweet']))
    print('Number of retweets:', number_of_retweets,'➝', (100*number_of_retweets)/number_of_tweets, '%')
    results.append({'Metric':'Number of retweets', 'Value':number_of_retweets})

    number_of_replies = int(eda.number_of_replies_query(text_column=fields['text_tweet']))
    print('Number of replies:', number_of_replies, '➝', (100*number_of_replies)/number_of_tweets, '%')
    results.append({'Metric':'Number of replies', 'Value':number_of_replies})

    most_active_users = eda.most_active_users_query(text_column=fields['text_tweet'], user_column=fields['user_username'], limit=10000)
    most_active_users.to_csv(results_path+'most_active_users.csv', index=False)
    print('Most active users detected! saved as '+results_path+'most_active_users.csv')

    most_visible_users = eda.most_visible_users_query(fields=fields, limit=10000)
    # print(most_visible_users)
    print('Most visible users detected! saved as '+results_path+'Visible_Users.csv')

    verified_users = eda.verified_users_query(username_column=fields['user_username'], verified_column=fields['user_verified'])
    # print(verified_users)
    print('Verified users detected! saved as '+results_path+'verified_users.csv')
    verified_users.to_csv(results_path+'verified_users.csv', index=False)

    # user_type = 'visible' # active, visible, verified
    # volume_of_tweets_per_user_type = eda.volume_of_tweets_per_user_type_query(username_column=fields['user_username'], user_type=user_type, date1=None, date2=None, 
    #                                                                           verified_column=fields['user_verified'], limit=100)
    # print(volume_of_tweets_per_user_type)

    country = 'United Kingdom'
    tweets_per_country_type = eda.tweets_per_country_query(country_column=fields['tweet_country'], country_code=country)
    tweets_per_country_type = int(tweets_per_country_type.values[0][0])
    print('Tweets per country ', country+':', tweets_per_country_type, '➝', (100*tweets_per_country_type)/number_of_tweets, '%')
    results.append({'Metric':'Number of tweets per country: '+country, 'Value':tweets_per_country_type})

    users_per_country = eda.users_per_country_query(username_column=fields['user_username'], country_column=fields['tweet_country'], country_code=country)
    print('Users per country', country+':', users_per_country.shape[0])
    results.append({'Metric':'Number of tweets per country: '+country, 'Value':users_per_country.shape[0]})

    language_code = 'en'
    tweets_per_language = int(eda.tweets_per_language_query(language_column=fields['tweet_language'], language_code=language_code))
    print('Tweets per language('+ language_code+'):', tweets_per_language, '➝', (100*tweets_per_language)/number_of_tweets, '%')
    results.append({'Metric':'Number of tweets per language: '+language_code, 'Value':tweets_per_language})

    volume_of_hashtags = eda.volume_of_hashtags_query(hashtag_column=fields['hashtags'])
    print('Number of hashtags:', volume_of_hashtags.shape[0])
    results.append({'Metric':'Number of hashtags', 'Value':volume_of_hashtags.shape[0]})
    
    df = eda.volume_of_hashtags_per_date_query(hashtag_column=fields['hashtags'], date_column=fields['tweet_date'])
    print('Number of hashtags per day calculated! saved as '+results_path+'number_of_hashtags_per_day.csv')
    df.to_csv(results_path+'number_of_hashtags_per_day.csv', index=False)
    
    number_tweets_featuring_url = eda.all_urls_query(url_column=fields['tweets_url'])
    number_tweets_featuring_url = int(number_tweets_featuring_url.shape[0])
    print('Number of tweets with URL:', number_tweets_featuring_url, '➝', (100*number_tweets_featuring_url)/number_of_tweets, '%')
    results.append({'Metric':'number of tweets with URL', 'Value':number_tweets_featuring_url})

    sources = eda.number_of_source_query(source_column=fields['tweet_source'])
    print('Number of data source:', sources.shape[0])
    results.append({'Metric':'Number of data source', 'Value':sources.shape[0]})

    tweets_per_source = eda.number_of_tweets_per_source_query(source_column=fields['tweet_source'])
    print('Number of tweets per source calculated! Saved as '+results_path+' number_of_tweets_per_source.csv')
    tweets_per_source.to_csv(results_path+'number_of_tweets_per_source.csv', index=False)

    ngrams = eda.run_n_gram(text_column=fields['text_tweet'], n=2, top_n=100, limit=False)
    eda.generate_wordcloud(ngrams)

    sentiment_results = eda.analyse_sentiment(tweet_id_column=fields['tweet_id'], text_column=fields['text_tweet'], limit=False)
    print('Sentiment analysis finished! saved as '+results_path+'sentiment_analysis.csv')
    sentiment_results.to_csv(results_path+'sentiment_analysis.csv', index=False)

    log_df = pd.DataFrame(results)
    log_df.to_csv(results_path+'log.csv', index=False)
    # eda.run_liwc(text_column=fields[1], dict_name='dicts_liwc/behavioral-activation-dictionary.dicx') #'''

    eda.peak_detection(3)

    # from liwc import Liwc
    # liwc = Liwc('dicts_liwc/behavioral-activation-dictionary.dicx')
    # # Search a word in the dictionary to find in which LIWC categories it belongs
    # print(liwc.search('happy'))

    # # Extract raw counts of words in a document that fall into the various LIWC categories
    # print(liwc.parse('I love ice cream.'.split(' ')))


    # y = df['num_hashtags'].values
    # x = np.array(range(len(y)))
    # # peak_detector = PeakDetector()
    # # print(peak_detector.find_peaks_lehmann(data))
    # # print(peak_detector.find_peaks_palshikar_s1(data))

    # # Find peaks
    # i_peaks, _ = find_peaks(y)

    # print(i_peaks)
    # # Find the index from the maximum peak
    # i_max_peak = i_peaks[np.argmax(y[i_peaks])]

    # # Find the x value from that index
    # x_max = x[i_max_peak]

    # print(type(x), type(y))

    # # Plot the figure
    # plt.plot(x, y)
    # plt.scatter(x=x[i_peaks], y=y[i_peaks], color='blue')
    # plt.axvline(x=x_max, ls='--', color="k")
    # plt.show()

    # plt.plot(data)
    # plt.plot(peaks, data[peaks], "x")
    # plt.plot(np.zeros_like(data), "--", color="gray")
    # plt.show() '''
