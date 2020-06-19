import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import operator
import math

def assign_count_values(row):
    file_name = row['file_names']
    folder_name = row['folder_names']
    for key, val in unique_words_by_filenames[file_name].items():
        if key in row.index:
            row[key] = val
    return row

def part1_load(class_1_folder, class_2_folder, n=1):
#     class_1_folder = "a1/grain"
#     class_2_folder = "a1/crude"
    class_1_df = pd.DataFrame(columns=['file_names', 'folder_names'])
    class_2_df = pd.DataFrame(columns=['file_names', 'folder_names'])
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    class_1_file_names = []
    class_2_file_names = []
    
    unique_words_by_filenames = {}
    complete_unique_words_list = []

    for file in os.listdir(class_1_folder):
        class_1_file_names.append(file)
        file_str = open(os.path.join(class_1_folder, file)).read().lower()
        file_words_list = tokenizer.tokenize(file_str) #remove punctuation and tokenise words
        unique_words = set(file_words_list)
        unique_words_with_counts = {}
        for words in unique_words : 
            complete_unique_words_list.append(words)
            unique_words_with_counts[words] = file_words_list.count(words)
        unique_words_by_filenames[file] = unique_words_with_counts
    class_1_df['file_names'] = class_1_file_names
    class_1_df['folder_names'] = class_1_folder

    for file in os.listdir(class_2_folder):
        class_2_file_names.append(file)
        file_str = open(os.path.join(class_2_folder, file)).read().lower()
        file_words_list = tokenizer.tokenize(file_str) #remove punctuation and tokenise words
        unique_words = set(file_words_list)
        unique_words_with_counts = {}
        for words in unique_words : 
            complete_unique_words_list.append(words)
            unique_words_with_counts[words] = file_words_list.count(words)
        unique_words_by_filenames[file] = unique_words_with_counts
    class_2_df['file_names'] = class_2_file_names
    class_2_df['folder_names'] = class_2_folder

    final_df = class_1_df.append(class_2_df, ignore_index=True)
#     print(final_df.head())
#     print("unique words by filenames dict\n\n\n\n")
#     print(unique_words_by_filenames)
#     print("total number of unique words in all files")
#     print(len(complete_unique_words_list))
    
    for word in complete_unique_words_list:
        final_df[word] = 0
    final_df = final_df.apply(assign_count_values, axis=1)
    
    return final_df

df = part1_load("a1/grain", "a1/crude")
df.head()

import operator

def part2_vis(df, m=5):
#     m = 5
    max_col_values = {}
    for col in df.columns:
        if col not in ['file_names', 'folder_names']:
            max_col_values[col] = max(df[col])

    max_col_values = dict(sorted(max_col_values.items(), key=operator.itemgetter(1), reverse=True)[:m])

    print(max_col_values)
    max_col_names = list(max_col_values.keys())
    print(list(max_col_values.keys()))
    
    max_grain_values = df[df['folder_names'] == 'a1/grain'][max_col_names].sum().values
    max_crude_values = df[df['folder_names'] == 'a1/crude'][max_col_names].sum().values
    
    new_df = pd.DataFrame({'grain':max_grain_values, 'crude':max_crude_values}, index = max_col_names)
    print(new_df)
    ax = new_df.plot.bar(rot=0)

plot = part2_vis(df)
df.head()

import math

def computeTF(wordDict, bowCount):
    tfDict = {}
    for word, count in wordDict.items():
        try:
            tfDict[word] = count/float(bowCount)
        except Exception as e:
            tfDict[word] = 0.0
    return tfDict

def computeIDF(docList,N):
    idfDict = {}
    for word, val in docList.items():
        try:
            idfDict[word] = math.log10(N / float(val))
        except Exception as e:
            idfDict[word] = 0.0
    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

def calculate_tf_idf(row,complete_df):
#     print(row)
    file_name = row['file_names']
    folder_name = row['folder_names']
    len_bow = row.drop(['file_names','folder_names']).astype(bool).sum()
    word_dict = row.drop(['file_names','folder_names']).to_dict()
    tfDict = computeTF(word_dict,len_bow)
#     print(tfDict)
    idfDict = computeIDF(word_dict, len(complete_df))
#     print(idfDict)
    tfidf = computeTFIDF(tfDict, idfDict)
    tfidf['file_names'] = file_name
    tfidf['folder_names'] = folder_name
#     print(tfidf)
#     tfidf_df.append(tfidf, ignore_index=True)
#     print(tfidf_df)
    return tfidf

def part3_tfidf(df):
    tfidf_df = pd.DataFrame(columns=df.columns)
    tfidf_list = df.apply(calculate_tf_idf, complete_df = df, axis=1).values
    tfidf_df = pd.DataFrame.from_records(tfidf_list)
    print("done")
    return tfidf_df

tfidf_df = part3_tfidf(df)

tfidf_df.head()

plot = part2_vis(tfidf_df)