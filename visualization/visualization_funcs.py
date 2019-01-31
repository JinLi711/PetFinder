"""
This file contains functions to help visualize the data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import os
import random


#======================================================================
# Set Up
#======================================================================


# paths
start_path = '../data'
breed_labels = start_path + '/breed_labels.csv'
color_labels = start_path + '/color_labels.csv'
state_labels = start_path + '/state_labels.csv'
data = start_path + '/train.csv'

# mappings
breed_labels_df = pd.read_csv(breed_labels)
color_labels_df = pd.read_csv(color_labels)
state_labels_df = pd.read_csv(state_labels)


#======================================================================
# Mapping Numerics To Categorical
#======================================================================


def df_to_dict(df, index, col_name):
    """
    Create a dictionary mapping numerics to categories.

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param index: The name of the index (the keys)
    :type  index: str
    :param col_name: The name of the column to be the values
    :type  col_name: str
    :returns: dictionary where keys are numerics and values are categories
    :rtype:   dict
    """

    df = df[[index, col_name]].set_index(index)
    map_dict = df.T.to_dict('list')

    for key, value in map_dict.items():
        map_dict[key] = value[0]

    return map_dict


# mappings
breed_labels_dict = df_to_dict(breed_labels_df, 'BreedID', 'BreedName')
color_labels_dict = df_to_dict(color_labels_df, 'ColorID', 'ColorName')
state_labels_dict = df_to_dict(state_labels_df, 'StateID', 'StateName')

col_cat_mappings = {
    'Type': {1: 'Dog', 2: 'Cat'},
    'Breed1': breed_labels_dict,
    'Breed2': breed_labels_dict,
    'Gender': {1: 'Male', 2: 'Female', 3: 'Mixed'},
    'Color1': color_labels_dict,
    'Color2': color_labels_dict,
    'Color3': color_labels_dict,
    'MaturitySize': {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'},
    'FurLength': {1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'},
    'Vaccinated': {1: 'Yes', 2: 'No', 3: 'Not Sure'},
    'Dewormed': {1: 'Yes', 2: 'No', 3: 'Not Sure'},
    'Sterilized': {1: 'Yes', 2: 'No', 3: 'Not Sure'},
    'Health': {1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 0: 'Not Specified'},
    'State': state_labels_dict,
}


def map_num_to_cat(df, col_cat_dict=col_cat_mappings):
    """
    For each appropriate column, map the numbers to the
    categories that it represents.

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param col_cat_dict: dictionary mapping columns to the mapping
    :type  col_cat_dict: dict
    :returns: Pandas dataframe
    :rtype:   pandas.core.frame.DataFrame
    """

    for col, mapping in col_cat_dict.items():
        df[col] = pd.Categorical(df[col]).rename_categories(mapping)

    return df


#======================================================================
# Visualizing Tabular Data Individually
#======================================================================


def bargraph(col, data):
    """
    Plot a bar graph of categorical data

    :param col: Column name
    :type  col: str
    :param data: Pandas dataframe
    :type  data: pandas.core.frame.DataFrame
    """

    count_types = data[col].value_counts()

    plt.figure(
        figsize=(24, 10)
    )
    sns.barplot(
        count_types.index,
        count_types.values,
        alpha=0.7
    )
    plt.title(
        col,
        fontsize=20
    )
    plt.ylabel(
        'Number of Occurrences',
        fontsize=14
    )
    plt.xlabel(
        'Types',
        fontsize=14
    )
    
    plt.show()
    
    
def head_value_counts(col, df):
    """
    Return the first 5 rows of the unique value counts of a certain column

    :param col: column name
    :type col: str
    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :returns: a series where the index are the first five unique values 
              and the values are the frequencies
    :rtype:   pandas.core.series.Series
    """

    head = df[col].value_counts()
    print(head.head())
    return head.head()


def histogram(label, data, title):
    """
    Create a histogram given a column from a dataframe.

    :param label: column name
    :type  label: str
    :param data: Pandas dataframe
    :type  data: pandas.core.frame.DataFrame
    :param title: title of histogram
    :type  title: str
    """

    plt.figure(
        figsize=(15, 10)
    )
    sns.distplot(data[label], kde=False, )
    plt.title(title)
    plt.show()
    plt.clf()
    
    
def num_unique_values(df):
    """
    Find the number of unique values of a dataframe.
    
    :param data: Pandas dataframe
    :type  data: pandas.core.frame.DataFrame
    """
    
    for col in df.columns:
        print(
            'col:',
            col,
            '\nNumber of Unique Values:',
            len(df[col].unique())

        )


def plot_inputs(df, bargraph_cols, histogram_cols, head_count_cols):
    """
    Plot different graphs depending on the data.
    Includes:
        bargraphs
        histograms

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param bargraph_cols: set of categories to be plotted on a bargraph
    :type  bargraph_cols: set
    :param histogram_cols: set of categories to be plotted on a histogram
    :type  histogram_cols: set
    :param head_count_cols: set of categories that are too large for histogram
    :type  head_count_cols: set
    """

    all_cols = set(df.columns)
    listed_cols = bargraph_cols.union(histogram_cols)
    other_cols = all_cols.difference(listed_cols).difference(head_count_cols)

    print('Bar Graphs:\n')
    for bargraph_col in bargraph_cols:
        print('\n', bargraph_col, '\n')
        bargraph(bargraph_col, df)

    print('Histograms:\n')
    for histogram_col in histogram_cols:
        print('\n', histogram_col, '\n')
        histogram(histogram_col, df, histogram_col)

    print('Head Counts:\n')
    for head_count_col in head_count_cols:
        print('\n', head_count_col, '\n')
        head_value_counts(head_count_col, df)

    print(
        '\nColumns Not Plotted:',
        other_cols
    )


#======================================================================
# Word Clouds
#======================================================================


def df_to_text(df, type_cat, cat):
    """
    Create a string of all the text in a column from a dataframe.

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param type_cat: Category to be split on
    :type  type_cat: str
    :param cat: Category to be considered
    :type  cat: str
    """

    items = df.loc[df['Type'] == type_cat, cat].values
    items = [str(item) for item in items]
    items = [item.replace(' ', '_') for item in items]
    return items


def generate_word_clouds(text, title):
    """
    Create a word cloud given text.

    :param text: text to be used for wordcloud
    :type  text: str
    :param title: title of the word cloud
    :type  title: str
    """

    wordcloud = WordCloud(
        max_font_size=None,
        background_color='white',
        width=1200,
        height=1000
    ).generate(text)

    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")


#======================================================================
# Visualizing Tabular Data With Respect To Labels
#======================================================================


def prepare_plot_dict(df, col, main_count):
    """
    Source: https://www.kaggle.com/artgor/exploration-of-data-step-by-step

    Preparing dictionary with data for plotting.
    Sshow how much higher/lower are the rates of Adoption speed 
    for the current column comparing to base values),
    Calculate base rates, then for each category in the column,
    calculate rates of Adoption speed and find difference with the base rates.

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param col: Column name to consider
    :type  col: str
    :param main_count: Initial starting count values for adoption speed
    :type  main_count: pandas.core.series.Series
    :returns: dictionary mapping a category to its comparison rate
    :rtype:   dict
    """

    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(
            df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = \
                    ((val_count[k] / sum(val_count.values())) /
                     main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0

    return plot_dict


def make_count_plot(
        df,
        x,
        hue='AdoptionSpeed',
        title='',
        main_count=None):
    """
    Source: https://www.kaggle.com/artgor/exploration-of-data-step-by-step

    Plotting countplot with correct annotations.

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param x: Column name to consider
    :type  x: str
    :param hue: Name of the label
    :type  hue: str
    :param title: title of the plot
    :type  title: str
    :param main_count: Initial starting count values for adoption speed
    :type  main_count: pandas.core.series.Series
    """

    g = sns.countplot(x=x, data=df, hue=hue)
    plt.title(f'AdoptionSpeed {title}')
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(
            text,
            (p.get_x() + p.get_width() / 2., h),
            ha='center',
            va='center',
            fontsize=11,
            color='green' if plot_dict[h] > 0 else 'red',
            rotation=0,
            xytext=(0, 10),
            textcoords='offset points'
        )

    
def create_label_plots(df, categories, main_count):
    """
    Create plots to compare adoption speed with respect to a feature.
    
    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param categories: List 
    :type  categories: list
    """
    
    for category in categories:
        plt.figure(figsize=(28, 18))
        make_count_plot(
            df=df, 
            x=category, 
            title='by pet {}'.format(category),
            main_count=main_count
        )


#======================================================================
# Visualizing Images
#======================================================================


train_dir = '../data/train_images/'
images = [i.split('-')[0] for i in os.listdir(train_dir)]


def show_images(data, trait1, trait2, images=images, train_dir=train_dir):
    """
    Show images with certain traits for both cats and dogs

    :param data: Pandas dataframe
    :type  data: pandas.core.frame.DataFrame
    :param trait1: category name of dataframe
    :type  trait1: str
    :param trait2: category name of dataframe
    :type  trait2: str
    :param images: list of image paths
    :type  images: list
    :param train_dir: train path
    :type  train_dir: str
    """

    # just cats and dogs
    for t in data['Type'].unique():
        for m in data[trait1].unique():
            df = data.loc[(data['Type'] == t) & (data[trait1] == m)]
            top = list(df[trait2].value_counts().index)[:5]
            print(f"Most common Breeds of {m} {t}s: (considering {trait2})")

            fig = plt.figure(figsize=(25, 4))

            for i, breed in enumerate(top):
                # excluding pets without pictures
                b_df = df.loc[(df[trait2] == breed) & (
                    df['PetID'].isin(images)), 'PetID']
                if len(b_df) > 1:
                    pet_id = b_df.values[1]
                else:
                    pet_id = b_df.values[0]
                ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

                im = Image.open(train_dir + pet_id + '-1.jpg')
                plt.imshow(im)
                ax.set_title(f'Breed: {breed}')
            plt.show()


#======================================================================
# Visualizing Text
#======================================================================


def view_text(df, col_name, num=5):
    """
    View some of the text with some context

    :param df: Pandas dataframe
    :type  df: pandas.core.frame.DataFrame
    :param col_name: name of the text column
    :type  col_name: str
    :param num: number of random posts to output
    :type  num: int
    """

    text = df[col_name]

    num_rows, num_cols = df.shape

    print("Sample Text: \n")
    for instance in range(num):
        randint = random.randint(0, num_rows)
        print(text[randint], '\n')


def avg_post_len(text):
    """
    Find the average text length of text in a series

    :param text: series of text
    :type  text: pandas.core.series.Series
    :returns: average text length
    :rtype:   float
    """

    def split_str(x):
        return str(x).split(' ')

    avg_length = text.apply(lambda x: len(split_str(x))).mean()

    return avg_length