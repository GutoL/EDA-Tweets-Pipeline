# EDA Tweets Pipeline
Project written in Python to perform exploratory analysis of data from tweets that are stored in Big Query. The project connects with a bigquery account and downloads previously collected content using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api). Thus, several analyzes are performed on the tweets.
To run the script, first install the libraries:

```console

pip install -r requirements.txt

```

Then you can run the script:

```console

python3 run_query.py setup.json

```

where the setup.json file contains the configuration needed to run the script (please, see the example in the project).