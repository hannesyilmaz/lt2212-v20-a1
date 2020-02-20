import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
# ADD ANY OTHER IMPORTS YOU LIKE

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.



def part1_load(folder1, folder2, n=1, m=10):
    # CHANGE WHATEVER YOU WANT *INSIDE* THIS FUNCTION.
    path_1 = [folder1]
    path_2 = [folder2]
    folder1_list = []
    folder2_list = []

    for index in path_1:
        for filename in os.listdir(index):
            with open(os.path.join(index, filename), 'r') as filedata:
                first_files = " ".join(filedata.read().split())
                
                d_1 = {}
                tokens = re.findall(r'\w+', first_files)
                for token in tokens:
                    if token in d_1:
                        d_1[token] = d_1[token] + 1
                    else:
                        d_1[token] = 1
                
                folder1_list.append(d_1)

    for i in path_2:
        for filename_2 in os.listdir(i):
            with open(os.path.join(i, filename_2), 'r') as filedata_2:
                second_files = " ".join(filedata_2.read().split())
                
                d_2 = {}
                tokens = re.findall(r'\w+', second_files)
                for token in tokens:
                    if token in d_2:
                        d_2[token] = d_2[token] + 1
                    else:
                        d_2[token] = 1
                
                folder2_list.append(d_2)
    
    folders = folder1_list, folder2_list
    

    name_of_the_files = []
    for files_a in path_1:
        for docs in os.listdir(files_a):
            name_of_the_files.append(docs)
    
    for files_b in path_2:
        for docus in os.listdir(files_b):
            name_of_the_files.append(docus)


    data_frame = pd.concat(map(pd.DataFrame, folders), axis=1).melt().dropna()
    
    data_frame.columns = ['word', 'frequency']

    return data_frame.nlargest(m, 'frequency')

    #return pd.DataFrame(npr.randn(2,2)) # DUMMY RETURN

df = part1_load("a1/crude/", "a1/grain/")

print(part1_load("a1/crude/", "a1/grain/"))

#print(part1_load("a1/crude/", "a1/grain/"))

def part2_vis(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    

    # CHANGE WHAT YOU WANT HERE
    #df = part1_load("a1/crude/")
    return df.plot(x = 'word', y='frequency', kind="bar")

print(part2_vis(df))


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
      
    def all_words(folder1, folder2):
        path_1 = [folder1]
        path_2 = [folder2]

        for index in path_1:
            for filename in os.listdir(index):
                with open(os.path.join(index, filename), 'r') as filedata:
                    first_files = " ".join(filedata.read().split())

        for i in path_2:
            for filename_2 in os.listdir(i):
                with open(os.path.join(i, filename_2), 'r') as filedata_2:
                    second_files = " ".join(filedata_2.read().split())


        all_files = first_files, second_files

        paragprahs = " ".join(all_files)

        tok_all = nltk.sent_tokenize(paragprahs)

        for i in range(len(tok_all)):
            tok_all[i] = tok_all[i].lower()
            tok_all[i] = re.sub(r'\W',' ',tok_all[i])
            tok_all[i] = re.sub(r'\s+',' ',tok_all[i])


        X = []
        for data in tok_all:
            vector = []
            for word in df['word'].tolist():
                if word in nltk.word_tokenize(data):
                    vector.append(1)
                else:
                    vector.append(0)
            X.append(vector)


        word_idfs = {}

        for word in df['word'].tolist():
            doc_count = 0
            for data in tok_all:
                data = nltk.word_tokenize(data)
                if word in data:
                    doc_count += 1
                    h = (len(tok_all) / doc_count )+1
                    word_idfs[word] = np.log(h)

                    
        tf_matrix = {}
        for word in df['word'].tolist():
            doc_tf = []
            for data in tok_all:
                frequency = 0
                data = nltk.word_tokenize(data)
                for w in data:
                    if w == word:
                        frequency += 1
                tf_word = frequency / len(data)
                doc_tf.append(tf_word)
        tf_matrix[word] = doc_tf



        tfidf_matrix = []
        for word in tf_matrix.keys():
            tfidf = []
            for value in tf_matrix[word]:
                score = value * word_idfs[word]
                tfidf.append(score)
        tfidf_matrix.append(tfidf)


        X = np.asarray(tfidf_matrix)
        X = np.transpose(X)

        print(X)


    all_words("a1/crude/", "a1/grain/")

print(part3_tfidf(df))

