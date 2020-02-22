from collections import defaultdict
import os

import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import constants as c


DATA_DIR = c.data_dir


def load_data(filename):

    return pd.read_csv(os.path.join(DATA_DIR, filename))


def plot_tweet_distribution(df):
    train_df = df
    tweet_lenghts = train_df.apply(lambda x: len(x.text), axis=1)
    tweet_lenghts_1 = train_df[train_df.target == 1].apply(lambda x: len(x.text), axis=1)
    tweet_lenghts_0 = train_df[train_df.target == 0].apply(lambda x: len(x.text), axis=1)

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3)

    n_nbins = 157

    # plt.figure(figsize=(20, 10))
    ax1.hist(tweet_lenghts, n_nbins)
    ax1.set_title("Overall tweet length distribution")
    ax2.hist(tweet_lenghts_0, n_nbins)
    ax2.set_title("Real tweet length distribution")
    ax3.hist(tweet_lenghts_1, n_nbins)
    ax3.set_title("Fake tweet length distribution")
    ax1.set_ylabel('# tweets')
    ax2.set_ylabel('# tweets')
    ax3.set_ylabel('# tweets')

    ax1.set_xlabel('tweet length')
    ax2.set_xlabel('tweet length')
    ax3.set_xlabel('tweet length')

    ax_hight = ax1.get_ylim()[1]

    ax1.text(10, ax_hight - 10, f"min length: " + str(np.min(tweet_lenghts)))
    ax1.text(10, ax_hight - 20, f"max length: " + str(np.max(tweet_lenghts)))
    ax1.text(10, ax_hight - 30, f"mean length: " + str(np.round(np.mean(tweet_lenghts))))
    ax1.text(10, ax_hight - 40, f"median length: " + str(np.median(tweet_lenghts)))

    ax_hight = ax2.get_ylim()[1]
    ax2.text(10, ax_hight - 10, f"min length: " + str(np.min(tweet_lenghts_0)))
    ax2.text(10, ax_hight - 20, f"max length: " + str(np.max(tweet_lenghts_0)))
    ax2.text(10, ax_hight - 30, f"mean length: " + str(np.round(np.mean(tweet_lenghts_0))))
    ax2.text(10, ax_hight - 40, f"median length: " + str(np.median(tweet_lenghts_0)))
    ax2.text(10, ax_hight - 50, f"example: " + train_df[train_df.target == 1].iloc[1]["text"])

    ax_hight = ax3.get_ylim()[1]
    ax3.text(10, ax_hight - 10, f"min length: " + str(np.min(tweet_lenghts_1)))
    ax3.text(10, ax_hight - 20, f"max length: " + str(np.max(tweet_lenghts_1)))
    ax3.text(10, ax_hight - 30, f"mean length: " + str(np.round(np.mean(tweet_lenghts_1))))
    ax3.text(10, ax_hight - 40, f"median length: " + str(np.median(tweet_lenghts_1)))
    ax3.text(10, ax_hight - 50, f"example: " + train_df[train_df.target == 0].iloc[1]["text"])

    plt.show()


def word_length_distribution(df):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_len = df[df['target'] == 1]['text'].str.split().map(lambda x: len(x))
    ax1.hist(tweet_len, color='blue')
    ax1.set_title('disaster tweets')
    tweet_len = df[df['target'] == 0]['text'].str.split().map(lambda x: len(x))
    ax2.hist(tweet_len, color='red')
    ax2.set_title('Not disaster tweets')
    fig.suptitle('Words in a tweet')
    plt.show()


def average_word_length_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    word = df[df['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='blue')
    ax1.set_title('disaster')
    word = df[df['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='red')
    ax2.set_title('Not disaster')
    fig.suptitle('Average word length in each tweet')


def word_cloud(corpus):

    words = ' '.join(corpus)
    words_wc = WordCloud(background_color='black',
                         max_font_size=160,
                         width=512, height=512).generate(words)
    plt.figure(figsize=(10, 8), facecolor="k")
    plt.imshow(words_wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


table = str.maketrans('', '', string.punctuation)
stop_words = stopwords.words('english')
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
porter = PorterStemmer()


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


def make_corpus(tweets):
    tokens = [" ".join(tknzr.tokenize(x)) for x in tweets]

    stripped = [w.translate(table).lower() for w in tokens]

    no_urls = [re.sub(r"http\S+", "", x) for x in stripped]

    no_html = [re.sub(r'<.*?>', "", x) for x in no_urls]

    no_emoji = [emoji_pattern.sub("", x) for x in no_html]

    stopped = [[w for w in sent.split() if not w in stop_words] for sent in no_emoji]

    stemmed = [[porter.stem(word) for word in tokens] for tokens in stopped]

    return [" ".join(words) for words in stemmed]


def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_bigrams(corpus):
    plt.figure(figsize=(16, 5))
    top_tweet_bigrams = get_top_tweet_bigrams(corpus)[0:20]
    x, y = map(list, zip(*top_tweet_bigrams))
    sns.barplot(x=y, y=x)
    plt.show()


def punctuation_dist(df):
    corpus_1_punct = [x for x in df[df['target'] == 1]["text"].str.split()]
    corpus_0_punct = [x for x in df[df['target'] == 0]["text"].str.split()]
    punctuation = string.punctuation

    plt.rcParams["figure.figsize"] = (17, 10)
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

    d1 = defaultdict(int)
    for s in corpus_0_punct:
        for i in s:
            if i in punctuation:
                d1[i] += 1

    r = sorted(d1.items(), key=lambda k: k[1], reverse=True)
    x, y = zip(*r)

    ax1.bar(x, y)
    ax1.set_title("Punctuation dist in non-disaster tweets")

    d2 = defaultdict(int)
    for s in corpus_1_punct:
        for i in s:
            if i in punctuation:
                d2[i] += 1
    r = sorted(d2.items(), key=lambda k: k[1], reverse=True)
    x, y = zip(*r)
    ax2.bar(x, y)
    ax2.set_title("Punctuation dist in disaster tweets")

