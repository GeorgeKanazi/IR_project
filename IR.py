import numpy as np
import pandas as pd
from inverted_index_gcp import *
import math

from numpy.linalg import norm # not used
from collections import Counter

def read_posting_list(inverted, bucket_name, w, folder_name):
    """
    Read posting list of word from bucket store
    :param  inverted: inverted index object
            name_bucket: name bucket that store the inverted index
            w: the requested word
    :return posting list - list of tuple (doc_id,tf)
    """
    with closing(MultiFileReader(bucket_name,folder_name)) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * 6)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * 6:i * 6 + 4], 'big')
            tf = int.from_bytes(b[i * 6 + 4:(i + 1) * 6], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

        

def get_OPT_Tfidf(q_tokens, index, bucket_name, folder_name, corpus_docs, N=100):
    """
    Tfidf Optimized function , recieves tokens as a list of tokens (with duplicates)
    :param q_tokens:  list of tokens , with duplicatse.
    :param index:  inverted index.
    :param corpus_docs : int , optimization - number of docs in corpus
    :param N : int, number of docs to return
    :return: list of documents : title , sorted by their IDF score with query
    """
    query_size = len(q_tokens)
    q_tokens = list(Counter(q_tokens).items())
    query_sim_dict = {}
    for q_word, query_word_count in q_tokens:
        if q_word in index.term_total.keys():
            for doc_id, word_count in read_posting_list(index, bucket_name, q_word, folder_name):
                if doc_id != 0:  # missing values
                    tw = word_count * np.log10(corpus_docs / index.df[q_word])
                    if doc_id in query_sim_dict.keys():
                        query_sim_dict[doc_id] += query_word_count * tw
                    else:
                        query_sim_dict[doc_id] = query_word_count * tw

    for doc_id in query_sim_dict.keys():
        query_sim_dict[doc_id] = query_sim_dict[doc_id]*(1/query_size) * (1/index.DL[doc_id])

    if len(query_sim_dict) > N:
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)[:N]
    else:  # return the whole thing
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)

def calc_BM25(q_tokens, index, bucket_name, folder_name, corpus_docs, avg_dl, k=3, b=0.25):
    q_tokens = list(Counter(q_tokens).items())
    query_sim_dict = {}
    for q_word, _ in q_tokens:
        if q_word in index.term_total.keys():
            q_idf = np.log((1 + corpus_docs) / (index.df[q_word] + 0.5))
            if q_word in index.term_total.keys():
                for doc_id, word_count in read_posting_list(index, bucket_name, q_word, folder_name):
                    if doc_id != 0:  # missing values
                        tw = word_count * (k + 1) / (word_count + k * (1 - b + b * index.DL[doc_id] / avg_dl))
                        if doc_id in query_sim_dict.keys():
                            query_sim_dict[doc_id] += tw * q_idf
                        else:
                            query_sim_dict[doc_id] = tw * q_idf
    return query_sim_dict

def get_opt_BM25(q_tokens, index, bucket_name, folder_name, corpus_docs, avg_dl, k=3, b=0.25, N=100):
    """
    BM25 retrieval model from inverted index .
    :param q_tokens - list of str,  tokens of words processed like corpus.
    :param index : inverted index class
    :param corpus_docs : int , optimization - number of docs in corpus
    :param avg_dl : float, average document size in corpus
    :param k : float, range between [1.2 ,2] optimized to corpus
    :param b : float , ~0.75 optimizable
    :param N : int , number of best-fit docs to retrieve
    :return: list of documents : title , sorted by their IDF score with query
    """
    ####
    # we found via optmization that the optimal parameter are:
    # k = 3
    # b = 0.25
    # we set those values as a default
    ####
    query_sim_dict = calc_BM25(q_tokens, index, bucket_name, folder_name, corpus_docs, avg_dl)

    if len(query_sim_dict) > N:
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)[:N]
    else:  # return the whole thing
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)



    


def get_opt_BM25_for_joint(q_tokens, index, corpus_docs, avg_dl, bucket_name, folder_name, k=3, b=0.25, N=100):
    """
    BM25 retrieval model from inverted index , intended for joining indices (returns tf score as well as doc)
    :param q_tokens - list of str,  tokens of words processed like corpus.
    :param index : inverted index class
    :param corpus_docs : int , optimization - number of docs in corpus
    :param avg_dl : float, average document size in corpus
    :param k : float, range between [1.2 ,2] optimized to corpus
    :param b : float , ~0.75 optimizable
    :param N : int , number of best-fit docs to retrieve
    :return: list of tokens : (title,tf-idf score) , sorted by their IDF score with query
    """
    
    query_sim_dict = calc_BM25(q_tokens, index, bucket_name, folder_name, corpus_docs, avg_dl)

    if len(query_sim_dict) > N:
        return [(key, query_sim_dict[key]) for key in sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)[:N]]
    else:  # return the whole thing
        return [(key, query_sim_dict[key]) for key in sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)]

def get_OPT_Cosine(q_tokens, index, bucket_name, folder_name, N=100):
    """
    Tfidf Optimized function , recieves tokens as a list of tokens (with duplicates)
    :param q_tokens:  list of tokens , with duplicatse.
    :param index:  inverted index.
    :param corpus_docs : int , optimization - number of docs in corpus
    :param N : int, number of docs to return
    :return: list of documents : title , sorted by their IDF score with query
    """
    query_size = len(q_tokens)
    query_counter = Counter(q_tokens)
    query_norm = norm([item[1]/query_size for item in query_counter.items()])
    q_tokens = list(query_counter.items())
    query_sim_dict = {}
    for q_word, query_word_count in q_tokens:
        if q_word in index.term_total.keys():
            for doc_id, word_count in read_posting_list(index, bucket_name, q_word, folder_name):
                if doc_id != 0:  # missing values
                    temp = word_count * query_word_count
                    if doc_id in query_sim_dict.keys():
                        query_sim_dict[doc_id] += temp
                    else:
                        query_sim_dict[doc_id] = temp

    for doc_id in query_sim_dict.keys():
        query_sim_dict[doc_id] = query_sim_dict[doc_id]*(1/query_norm) * (1/index.docs_norm[doc_id])

    if len(query_sim_dict) > N:
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)[:N]
    else:  # return the whole thing
        return sorted(query_sim_dict, key=query_sim_dict.get, reverse=True)