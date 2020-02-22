# Real or Not? NLP with Disaster Tweets 

Sentence classification is an important and common task for machine learning models with application in sentiment analysys and detection of email spam. This project trains models
to distinguish real disaster tweets from tweets that although using disaster-like language, do not relate to real disasters (e.g. "my day is a train wreck so far"). It evaluates the
performance of simple probabilistic models (naive bayes), more advanced models using neural networks and word embeddings, as well as the latest cutting-edge language models (e.g. bert). 

The dataset used, a collection of 10k tweets, is provided by this [kaggle tutorial](https://www.kaggle.com/c/nlp-getting-started), and was initially compiled by [figure-eight](https://www.figure-eight.com/data-for-everyone/).

Follow these notebook in order:

- [EDA](): Performs exploratory data analysis of the dataset
- [naive baeyes models](): This notebooks used naive bayes models to classify the tweets as pertaining to real diasters or not. This are simple probabilistic models that were popularized in spam classification of emails. They do a remarkable good job, 
achieving a accuracy of xx%. It uses both the scikit implementation, as well as a implementation from scratch. Additionaly, it compares a couple of different ways to represent the tweets (bag of words, TF-IDF)
- simple [neural network with word embeddings] - this notebook explores using a neural netweork, using the [GloVe](https://nlp.stanford.edu/projects/glove/) word representation.
 While bag of words and TF-IDF representations rely on word frequencies, these models' representation are learned from a unsupervised learning algorithm (a shallow neural net in the case of word2vec, and a log-bilinear regression model on a co-occurrence matrix in the case of GloVe), and are
based on word co-occurrence (their linguistic context) - they capture the semantics of analogic. The first of such algorithms was word2vec, using a continuous bag of words, and ski-grams. These models can be used to find similarities between words (king is to queen as  man is to woman), or to generate word embeddings to train other models.
- [Bert]: this notebook uses the current state of the art language model, [bert](http://jalammar.github.io/illustrated-bert/). This, and other models in the same family, are based on the transformer model and the concept of attention. This mechanism considers the relative importance of words in a sentence when training the model. It uses [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) to adapt
the pre-trained bert model to our specific usage. The first of this fancy models was bert, which builds on word embeddings by considering context. With word2vec and GloVe, a word would have
the same vector no matter the context. Elmo models consider context, so the same word will have multiple embeddings depending on the context. Elmo works by learning to predict the next word in a sentence (remember word2vec and GloVe learned similarity and analogy).



These models achieved accuracies of xx, xxx and xx for scikit naive bayes, homebrew naive bayes, word embeddings, elmo, and bert, respectively

## A word of metrics
Above, we used "accuracy" as a metric to compare the different models. This is an intuitive metric to understand: it means how accurate the model is, that is, how often is it correct.
But that is hardly the only measure, and not always the most appropriate.
# TODO: cancer example



just one possible


Fun, now what can we do with these models? After all, a model is only useful if it's actually used witha purpose - that is, it is put into production. One usage is to monitor tweeter for disasters, and improve our model in real time.
The script [x]() launches the infrastrure into a k8s cluster to deploy a tweeter monitoring and training system.
# TODO: lambda architecture here

- tweets and monitored. a dashboard displays the most recent disaster tweets.
- all tweets are evaluated for acuracy. and this input is fed back into the system for online training. This required a ground-truth classification of tweets. Ideally a human would make this decision. However, we want to minimize requirement of human intervention in our system - that would be laborious. Instead, we'll accept non-disaster classification as true. and we'll compare positive classifications agains gldelt for validation.







 



