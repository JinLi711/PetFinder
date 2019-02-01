"""
Module for preprocessing the data.

"""


"""
OHE: One Hot Encoding
STD: Standard Scaler

What I did for each category:
'Type'          OHE
'Name'          Binary binning, OHE
'Age'           STD
'Breed1'        OHE
'Breed2'        replace 0 with None, OHE
'Gender'        OHE
'Color1'        OHE
'Color2'        replace 0 with None, OHE
'Color3'        replace 0 with None, OHE
'MaturitySize'  OHE
'FurLength'     OHE
'Vaccinated'    OHE
'Dewormed'      OHE
'Sterilized'    OHE
'Health'        OHE
'Quantity'      STD
'Fee'           STD
'State'         OHE
'RescuerID'     Drop
'VideoAmt'      STD
'Description'   Drop
'PetID'         
'PhotoAmt'      STD
"""


import os

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

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


#======================================================================
# Preprocess Tabular Data
#======================================================================


def name_cat(data, col_name):
    """
    Deal with the category "Name".
    Observations:
        Pets with less than 4 characters in their names
        most likely aren't real names.
        Some of the names aren't really names (ex. kitten, girl, cat)
        There can be multiple names if the instance is refering
        to a group of animals.
        Some names really imply no name. (ex. No Name)
    My plan:
        binary conversion to have name or does not.
        Less than 4 characters will be considered no name.
        NaN: no name.
        If the name is something like ("No Name"), 
        consider it as a name anyways.
        If has generic name, consider it as has a name anyways.
        If the name is a number, say it does not have a name

    :param data: dataframe
    :type  data: pandas.core.frame.DataFrame
    :param col_name: name of the column name
    :type  col_name: str
    :returns:  dataframe
    :rtype:   pandas.core.frame.DataFrame
    """

    df = data.copy()

    def name_or_not(name):
        """
        Check if it is a name or not.

        :param name: input for the name
        :type  name: str
        :returns: "Name" if it is, "No_Name" if not
        :rtype:   str
        """

        if type(name) is float:
            return "No_Name"
        if (len(name) < 4):
            return "No_Name"
        else:
            return "Name"

    values = {col_name: 0}

    df.fillna(value=values)

    names = df[col_name]
    names = names.apply(lambda x: name_or_not(x))
    df[col_name] = names

    return df


def replace_values(data, col_names, before, after):
    """
        For each appropriate column, 
        replace a certain value with another.

        :param data: dataframe
        :type  data: pandas.core.frame.DataFrame
        :param col_name: name of the column
        :type  col_name: str 
        :param before: anything to be replaced
        :type  before: int, str
        :param after: what it is replaced with
        :type  after: int, str
        :returns: dataframe
        :rtype:   pandas.core.frame.DataFrame
    """

    df = data.copy()

    values = {}
    for col_name in col_names:
        values[col_name] = before
    df.replace(values, after, inplace=True)

    return df


def clean_df(
        data,
        name_col,
        replace_val_cols,
        before, after,
        drop_cols):
    """
    :param data: dataframe
    :type  data: pandas.core.frame.DataFrame
    :param name_col: name of the name column
    :type  name_col: str
    :param replace_val_cols: list of columns to replace values
    :type  replace_val_cols: list
    :param before: anything to be replaced
    :type  before: int, str
    :param after: what it is replaced with
    :type  after: int, str
    :param drop_cols: list of columns to drop
    :type  drop_cols: list
    :returns: dataframe
    :rtype:   pandas.core.frame.DataFrame
    """

    df = name_cat(data, name_col)
    df = replace_values(df, replace_val_cols, before, after)
    df.drop(drop_cols, axis=1, inplace=True)

    return df


class DataCleaning(BaseEstimator, TransformerMixin):
    """
    Class for performing data cleaning, 
    so the same can be applied to the valid/test case
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :param X: dataframe
        :type  X: pandas.core.frame.DataFrame
        :param y: series
        :type  y: pandas.core.series.Series
        """
        return self

    def transform(self, X, y=None):
        X_trans = clean_df(
            X,
            'Name',
            ['Breed1', 'Breed2', "Color1", "Color2", "Color3"],
            0, "None",
            ['RescuerID', 'Description']
        )
        X_trans = rm.convert_obj_columns_to_cat(X_trans, {'PetID'})
        return X_trans


def split_cat_num(df):
    """
    Split the dataframe into two dataframes, 
    where one contains the numeric datatypes 
    and the other contains categorical datatypes.

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :returns: (df with numeric dtypes, 
               df with categorical dtypes)
    :rtype:   (pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame)
    """

    col_names = set(df.columns)
    types = set([str(dtype) for dtype in df.dtypes.values])

    num_cols = df.select_dtypes(
        include=['int8', 'int16', 'int32', 'int64', 'float64'])
    cat_cols = df.select_dtypes(include=['category'])

    num_col_names = set(num_cols.columns)
    cat_col_names = set(cat_cols.columns)

    missing_col_names = col_names.difference(
        num_col_names).difference(cat_col_names)

    if len(missing_col_names) != 0:
        print("Columns Missing:", missing_col_names)

    return num_cols, cat_cols


def transform_df(df, cat_encode_type):
    """
    Convert the dataframe to one hot for categorical,
    standard scaler for numerics.
    
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :param cat_encode_type: The type of encoding (one hot or just label)
    :type  cat_encode_type: str 
    :returns: (numpy array of size (instances, 
              number of numeric columns + category one hot encoded columns),
              pipeline)
    :rtype:   (numpy.ndarray,
               sklearn.compose._column_transformer.ColumnTransformer)
    """

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    num_cols, cat_cols = split_cat_num(df)
    num_col_names = list(num_cols.columns)
    cat_col_names = list(cat_cols.columns)

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    
    if cat_encode_type == "one hot":
        # one hot encoded: a lot more columns
        from sklearn.preprocessing import OneHotEncoder
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_col_names),
        ("cat", OneHotEncoder(), cat_col_names),
        ])
        result = full_pipeline.fit_transform(df).toarray()
        
    elif cat_encode_type == "numeric":
        from sklearn.preprocessing import OrdinalEncoder
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_col_names),
        ("cat", OrdinalEncoder(), cat_col_names),
        ])
        result = full_pipeline.fit_transform(df)

    else:
        raise ValueError("Not an available encoder type")

    return result, full_pipeline