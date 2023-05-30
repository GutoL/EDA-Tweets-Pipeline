from utils.main_functions import pipeline


if __name__ == "__main__":
    # fields = ['id', 'body', 'actor_id', 'preferredUsername', 'displayName']

    # fields = {
    #             'tweet_id': 'id',
    #             'tweet_date' : 'postedTime',
    #             'text_tweet': 'body',  
    #             'user_id': 'actor_id', 
    #             'user_username': 'preferredUsername', 
    #             'user_name': 'displayName',
    #             'tweet_country': 'country_code',
    #             'tweet_language': 'twitter_lang',
    #             'hashtags': 'hashtags',
    #             'tweet_source': 'generator_link',
    #             'tweets_url': 'url',
    #             'user_verified':'verified'
    #         }
    
    fields = {
                'tweet_id': 'id',
                'tweet_date' : 'created_at',
                'text_tweet': 'text',  
                'user_id': 'author_id', 
                'user_username': 'author_username', 
                'user_name': 'author_location',
                'tweet_country': 'geo_country',
                'tweet_language': 'lang',
                'hashtags': 'entities_hashtags',
                'tweet_source': 'source',
                'tweets_url': 'entities_urls',
                'user_verified':'author_verified'
            }

    # fields = ['twitter.id', 'twitter.text', 'twitter.user.id', 'twitter.user.screen_name', 'twitter.user.location']
    # fields = ['id', 'text', 'user_id_str', 'user_location', 'user_screen_name']

    pipeline('setup.json', fields, 'results/')