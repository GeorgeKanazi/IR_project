# IR_project
Information Retrieval Project 2023

## Wikipedia Search Engine
Final project in Information retrival Course 372.1.4406 at Ben-Gurion University, submitted by [myself](https://github.com/GeorgeKanazi) and Amar.

## Introduction
The engine is BM25 based. It uses various inverted indices on wikipedia (body text, title, and anchor text) as well as page rank, page view and page title dictionaries.
We've examined different retrival models (TF-IDF, BM-25, Binary Ranking) and combination of indices (title and body).

## Main Components
**search_frontend.py**- code given by the course staff. Contains a Flask interface to run on server, which inserts queries to retrive via 5 functions : 
_Search_ - Our main retrival model.
_Search_body_ - BM25 model with optimized values, only on body.
_Search_title_ /_search_anchor_ - Two methods to conduct a binary search for terms in the anchor \ title indices.
_get_pageRank_ , _get_Pageview_ - Two methods to return the PR or PV values of a given list of Doc-id's.

Overall, we've done nothing in this file besides connection our functions to it, reading the indicies and a few calculationsto reduce in-query computations.

**Retrev.py**-  Our main retrivers. Contains the functions that are called from the frontend.
_Corpus_tokenizer_ for query processing that matches the corpus (index). currently only has a default implementation, we were hoping to add stemming and lemmatization but didn't manage to in time.
_get_Binary_ - for binary retrival on indices (for the requested title/anchor text retrival).
_text_title_merge_ - A functions that runs BM25 queries on both the title index and text index, re-sorts the combined results according to retrived BM25 scores and weights.
We've found that the weights 0.7 for the body and 0.3 for the title works best on our corpus.
_get_TFIDF_ - Returns values accoring to TF_IDF calculation, BM25, and cosine similarity. has three PIPE's - 'HW' for the requested search_body implementation ,
 'opt' for a BM25 retrival with optimizied parameters (k1 ,b) over the wikipedia corpus -tested on supplied test queries, and 'cos' which uses cosine similarity on tfidf scores.

**IR.py**- Where most of the calculations happen.  contains 6 functions , they are pretty similar.
_read_posting_list_ - reads posting lists from the bucket
_get_OPT_Tfidf_ - retrives a list of docs according to tokens, uses sequencial implementation instead of the naive-vectoric model.
_get_BM25_ - same, only using BM25 model and default k1 , b parameters of (1.2 , 0.5)
_get_opt_BM25_ - we found different parameters that gave us improved results (k1 =3, b =0.25)
_get_opt_BM25_for_joint_ - returns a list of (doc_id , BM25 score) for further sorting and merging.
_get_OPT_Cosine_ - returns the cosine similarity based on the tf-idf score

**inverted_index_class.py** - class for inverted index object, contains document length,normalized document length, total_terms and posting locs dictionaries. Capble of reading and writing posting locs to and from bins.
