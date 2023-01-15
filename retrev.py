from inverted_index_gcp import *
import nltk
from nltk.tokenize import word_tokenize  # use is commented
from IR import *
import re
# should be activated only one time !
# nltk.download('stopwords')

from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))

# FROM GCP FOR THE SEARCH BODY PART
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


def Corpus_Tokenizer(text, method='norm'):
    """"
    During developing - had a 'stem' and 'lemm' methods. # CHANGE
    """

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    if method == 'norm':
        return [token for token in tokens if token not in all_stopwords]


def get_binary(query, inv_idx, bucket_name, folder_name):
    """
    Function to binary search in indices (for either anchor or title retrieval)
    :param query:  query to tokenize and search
    :param inv_idx: inverted index to binary search in
    :return: list of relevant documents  , sorted by query tokens matches
    """
    # tokens = word_tokenize(query.lower())
    tokens = Corpus_Tokenizer(query)
    ret = {}
    for tok in tokens:
        try:
            res = read_posting_list(inv_idx, bucket_name, tok, folder_name)
            for doc_id, _ in res:
                try:
                    ret[doc_id] += 1
                except:
                    ret[doc_id] = 1
        except Exception as e:
            print('Error in Anchor/Title index occurred - ', e)
    return sorted(ret, key=ret.get, reverse=True)


def text_title_Merge(query, text_idx, text_n_docs, text_avg_doc, title_idx, title_n_docs, title_avg_doc, N=200):
    """
    receives a query and runs it on text and title indices.
    returns a merged list of docs - based on TFIDF score of each index and pre-determined weights.
    :param query: string, query to tokenize and search.
    :param text_idx : inverted index of text.
    :param text_n_docs : int, number of docs in text
    :param text_avg_doc : float , average length of doc in text index.
    :param title_idx : inverted index of title
    :param title_n_docs : int, number of docs in title.
    :param title_avg_doc : float , average length of title docs.
    :param N : int, amount to return
    :returns : list of doc_id . sorted by combined tf-idf scores and weights.
    """
    tex_w = 0.74  # found via optimization
    tit_w = 0.31  # found via optimization
    bucket_name = "206549784"
    folder_text = "postingText"
    folder_title = "postings_title"

    query_tokens = Corpus_Tokenizer(query)
    text_retrieval = get_opt_BM25_for_joint(query_tokens, text_idx, text_n_docs, text_avg_doc, bucket_name, folder_text, N=N)
    title_retrieval = get_opt_BM25_for_joint(query_tokens, title_idx, title_n_docs, title_avg_doc, bucket_name, folder_title, N=N)
    doc_dict = {}
    for doc, score in text_retrieval:
        doc_dict[doc] = tex_w * score
    for doc, score in title_retrieval:
        if doc not in doc_dict.keys():
            doc_dict[doc] = tit_w * score
        else:
            doc_dict[doc] += tit_w * score
    query_result = sorted(doc_dict, key=doc_dict.get, reverse=True)
    return query_result


def get_TFIDF(q_text, index, corpus_docs, avg_dl, bucket_name, folder_name, N=100, PIPE='HW'):
    """
    Function that retrieves top N files matching each query according to TFIDF and cosine similarity.
    :param q_text: free text of query
    :param index: inverted index to search in
    :param N: top number of documents to retrieve
    :param corpus_docs : int , optimization - number of docs in corpus
    :param avg_dl : float, optimization - average document size in corpus
    :param PIPE: differentiate between naive (homework pipe and optimized)
    :return: list of docs id, sorted by rank
    """
    # preprocess according to corpus preprocess
    q_tokens = list(set(Corpus_Tokenizer(q_text)))

    # retrieve docs and score

    if PIPE == 'HW':
        # HW expects queries as dictionary of {id  : tokens }
        # res = pipe1.get_topN_score_for_queries({1: q_tokens}, index, N)[1]
        ret = get_OPT_Tfidf(q_tokens, index, bucket_name, folder_name, corpus_docs, N)
        return ret

    if PIPE == 'opt':
        # using optimized tfIDF
        ret = get_opt_BM25(q_tokens, index, bucket_name, folder_name, corpus_docs, avg_dl, N)
        return ret

    if PIPE == 'cos':
        # using cosine similarity
        ret = get_OPT_Cosine(q_tokens, index, bucket_name, folder_name, N=100)
        return ret



