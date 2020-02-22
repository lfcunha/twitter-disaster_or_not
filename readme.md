# Real or Not? NLP with Disaster Tweets 

Sentence classification is an important and common task for machine learning models with application in sentiment analysys and detection of email spam. This project trains models
to distinguish real disaster tweets from tweets that although using disaster-like language, do not relate to real disasters (e.g. "my day is a train wreck so far"). It evaluates the
performance of simple probabilistic models (naive bayes), more advanced models using neural networks and word embeddings, as well as the latest cutting-edge language models (e.g. bert). 

The dataset used, a collection of 10k tweets, is provided by this [kaggle tutorial](https://www.kaggle.com/c/nlp-getting-started), and was initially compiled by [figure-eight](https://www.figure-eight.com/data-for-everyone/).

Follow these notebook in order:

- [EDA](https://github.com/lfcunha/twitter-disaster_or_not/blob/master/notebooks/twitter-disaster_or_not/notebooks/EDA.ipynb): Performs exploratory data analysis of the dataset
- [naive baeyes models](https://github.com/lfcunha/twitter-disaster_or_not/blob/master/notebooks/twitter-disaster_or_not/notebooks/naive_bayes.ipynb): This notebooks used naive bayes models to classify the tweets as pertaining to real diasters or not. This are simple probabilistic models that were popularized in spam classification of emails. They do a remarkable good job, 
achieving a accuracy of xx%. It uses both the scikit implementation, as well as a implementation from scratch. Additionaly, it compares a couple of different ways to represent the tweets (bag of words, TF-IDF)
- [ConvID and LSTM with word embeddings](https://github.com/lfcunha/twitter-disaster_or_not/blob/master/notebooks/embeddings+LSTM.ipynb) - this notebook explores using a neural netweork, using the [GloVe](https://nlp.stanford.edu/projects/glove/) word representation.
 While bag of words and TF-IDF representations rely on word frequencies, these models' representation are learned from a unsupervised learning algorithm (a shallow neural net in the case of word2vec, and a log-bilinear regression model on a co-occurrence matrix in the case of GloVe), and are
based on word co-occurrence (their linguistic context) - they capture the semantics of analogic. The first of such algorithms was word2vec, using a continuous bag of words, and ski-grams. These models can be used to find similarities between words (king is to queen as  man is to woman), or to generate word embeddings to train other models.
- [Bert](https://github.com/lfcunha/twitter-disaster_or_not/blob/master/notebooks/bert.ipynb): this notebook uses the current state of the art language model, [bert](http://jalammar.github.io/illustrated-bert/). This, and other models in the same family, are based on the transformer model and the concept of attention. This mechanism considers the relative importance of words in a sentence when training the model. It uses [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) to adapt
the pre-trained bert model to our specific usage. The first of this fancy models was bert, which builds on word embeddings by considering context. With word2vec and GloVe, a word would have
the same vector no matter the context. Elmo models consider context, so the same word will have multiple embeddings depending on the context. Elmo works by learning to predict the next word in a sentence (remember word2vec and GloVe learned similarity and analogy).


This works builds on the ideas of several kaggle notebooks
- https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
- https://www.kaggle.com/mehulkumar99/keras-word-embeddings-lstms
- https://www.kaggle.com/c/nlp-getting-started/discussion/128563
- https://www.kaggle.com/rahulvks/text-rank-sub-keyword-analysis-nlu
- https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
- https://www.kaggle.com/mehulkumar99/disaster-tweets-minimal-word-level-vanilla-rnn
- https://www.kaggle.com/apjansing/bow-for-disaster-detection
- https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
- https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline


## Metrics
We've used several metrics to compare the models. The kaggle competition uses the F1 score. 
Accuracy: This is an intuitive metric to understand: it means how accurate the model is, that is, how often is it correct. But that is hardly the only measure, and not always the most appropriate.
Precision: fraction of relevant documents among the ones that were classified as relevant
Recall: fraction of relevant documents retrieved from the pool of relevant documents in the dataset
F1 score: The hamronic mean between precision and recall.
ROC curve (and AUC measure): plot of True positive rate vs false positive rate. random model has a AUC = 0.5. The closer to 1 the closer the model (means model classifies all pos without making mistakes.) If the AUC is below 0.5, your model is worse than random and something is wrong.
Precision/recall curve: Plots how the precision is impacted as the model recalls more relevant documents (it will be less precise with higher recall)


These models achieved accuracies of 0.77, 0.78, xx and xx for scikit naive bayes, LSTM/GloVe, bert, and CPT-2 respectively, on the test set (note that the true labels of the test set were obtained from the original dataset from [figure eight](https://www.figure-eight.com/data-for-everyone/).

The performance using a Naive Bayes model and a LSTM with word embedings is very similar. It's likely that we don't have enough data to train the LSTM model. In fact, the 
training plots of accuracy and loss of train vs valition sets shows the model is very overfit.


### Resources:
- https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html
- https://medium.com/mlreview/understanding-building-blocks-of-ulmfit-818d3775325b


# TODO: lambda architecture here

- tweets and monitored. a dashboard displays the most recent disaster tweets.
- all tweets are evaluated for acuracy. and this input is fed back into the system for online training. This required a ground-truth classification of tweets. Ideally a human would make this decision. However, we want to minimize requirement of human intervention in our system - that would be laborious. Instead, we'll accept non-disaster classification as true. and we'll compare positive classifications agains gldelt for validation.


