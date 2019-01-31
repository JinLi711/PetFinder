"""
Module for preprocessing the data.

"""


import os

import pandas as pd
import numpy as np

import reduce_memory as rm
import visualization_funcs as vf


#======================================================================
# Get the data.
#======================================================================


def get_data(data_path, test_path, label_name, valid_size=0.2):
    """
    Get the data, and split it into train and validation.
    Also, downcast numeric labels.

    :param data_path: path of the data input
    :type  data_path: str
    :param test_path: path of the test
    :type  test_path: str
    :param label_name: column name for label
    :type  label_name: str
    :param valid_size: Proportion to split on for the validation size
    :type  valid_size: float
    :returns: 5 dataframes
    :rtype:   (pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame) 
    """

    from sklearn.model_selection import train_test_split

    data = pd.read_csv(data_path)
    data = vf.map_num_to_cat(data)
    data = rm.downcast_df_int_columns(data)

    labels = data[label_name]
    data.drop(label_name, axis=1, inplace=True)

    test = pd.read_csv(test_path)
    test = vf.map_num_to_cat(test)
    test = rm.downcast_df_int_columns(test)

    X_train, X_valid, y_train, y_valid = train_test_split(
        data,
        labels,
        test_size=valid_size,
        random_state=42
    )

    return X_train, X_valid, y_train, y_valid, test


#======================================================================
# Preprocess Text
#======================================================================


def clean_text(text):
    """
    Clean the text

    :param text: the comments of one person
    :type  text: str
    :return: string of cleaned words
    :rtype:  str
    """

    text = text.lower()

    # remove ...
    text = re.sub(r'[.]+', r'.', text)
    # replace all numbers with <NUM>
    text = re.sub(r"[0-9]+", r" <NUM> ", text)
    # remove all excessive white spaces
    text = re.sub(r' +', r' ', text)
    return text


def clean_posts(df, col_name):
    """
    Clean each posts in the dataframe.

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """

    df['Cleaned_Posts'] = df[col_name].apply(
        lambda x: clean_text(x)
    )
    return df


def get_GloVe(glove_dir):
    """
    Open Stanford's GloVe file with 100 dimensional embeddings.
    The folder is downloaded from:
        https://nlp.stanford.edu/projects/glove/
    and is unmodified.

    :param glove_dir: directory of the glove file
    :type  glove_dir: str
    :return: dictionary where the keys are the words, 
             and values are the 100d representation
    :rtype:  dict
    """

    # dictionary that maps words into 100d array
    embeddings_index = {}
    file = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()
    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def map_words_to_int(cleaned_posts, max_words, maxlen):
    """
    Create a mapping from words to integer representation

    :param cleaned_posts: a 1-dim array of posts
    :type  cleaned_posts: numpy.ndarray
    :param max_words: maximum amount of unique words 
                      in the embedding vector space
    :type  max_words: int
    :param maxlen: maximum number of words considered for each instance. 
                   The rest of the post is cut off.
    :type  maxlen: int
    :returns: (Numpy array of (samples, maxlen) ,  
               dictionary where keys are words and
               values are the integer representation)
    :rtype:   (numpy.ndarray, dict)
    """

    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(cleaned_posts)

    """
    sequences is a list of lists,
    where each item of the outer list is an list of words
    in integer representation
    """
    sequences = tokenizer.texts_to_sequences(cleaned_posts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    from keras.preprocessing.sequence import pad_sequences
    # turns the lists of integers into a
    # 2D integer tensor of shape (samples, maxlen)
    sequences = pad_sequences(sequences, maxlen=maxlen)

    return (sequences, word_index)


def create_embedding_matrix(
    word_index,
    embeddings_index,
    max_words,
    embedding_dim
):
    """
    :param word_index: dictionary where keys are words and
                       values are the integer representation
    :type  word_index: dict
    :param embeddings_index: dictionary where the keys are the words, 
                             and values are the 100d representation
    :type  embeddings_index: dict
    :param max_words: maximum amount of unique words 
                      in the embedding vector space
    :type  max_words: int
    :param embedding_dim: number of dimensions that each word is mapped to
    :type  embedding_dim: int
    :returns: an array of shape (max_words, embedding_dim)
    :rtype:   numpy.ndarray
    """

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # index 0 is suppose to just be a placeholder
                embedding_matrix[i] = embedding_vector
    return embedding_matrix