import os
import pickle
import random

import lightgbm as lgb
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from scipy.spatial.distance import cosine

from .bsbi import BSBIIndex
from .compression import VBEPostings


def load_documents(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id, content = line.strip().split(" ", 1)
            documents[doc_id] = content.split()
    return documents


def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            q_id, content = line.strip().split(" ", 1)
            queries[q_id] = content.split()
    return queries

def load_relevance_judgments(file_path, queries, documents):
    q_docs_rel = {}  # Grouping by q_id
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            q_id, doc_id, rel = line.strip().split(" ")
            if q_id in queries and doc_id in documents:
                if q_id not in q_docs_rel:
                    q_docs_rel[q_id] = []
                q_docs_rel[q_id].append((doc_id, int(rel)))
    return q_docs_rel


def create_dataset(queries, documents, q_docs_rel):
    NUM_NEGATIVES = 1
    group_qid_count = []
    dataset = []

    for q_id, docs_rels in q_docs_rel.items():
        group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
        for doc_id, rel in docs_rels:
            dataset.append((queries[q_id], documents[doc_id], rel))
        # Menambahkan negative sample
        negative_sample = random.choice(list(documents.values()))
        dataset.append((queries[q_id], negative_sample, 0))

    return dataset, group_qid_count

def create_lsi_model(documents, num_topics=200):
    dictionary = Dictionary(documents.values())
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents.values()]
    lsi_model = LsiModel(bow_corpus, num_topics=num_topics)
    return lsi_model, dictionary
def vector_representation(text, lsi_model, dictionary):
    bow = dictionary.doc2bow(text)
    lsi_vector = lsi_model[bow]
    return [score for _, score in lsi_vector]

def calculate_features(query_vector, doc_vector, query, doc):
    cosine_dist = cosine(query_vector, doc_vector)
    jaccard = len(set(query) & set(doc)) / len(set(query) | set(doc))
    return [cosine_dist, jaccard]

# Fungsi untuk membuat fitur gabungan query dan dokumen
def create_feature_vectors(dataset, lsi_model, dictionary):
    X = []
    Y = []
    for query, doc, rel in dataset:
        query_vector = vector_representation(query, lsi_model, dictionary)
        doc_vector = vector_representation(doc, lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query, doc)
        X.append(features)
        Y.append(rel)
    return np.array(X), np.array(Y)

def predict_ranking(query, docs, ranker, lsi_model, dictionary):
    X_unseen = []
    for doc_id, doc in docs:
        query_vector = vector_representation(query.split(), lsi_model, dictionary)
        doc_vector = vector_representation(doc.split(), lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query.split(), doc.split())
        X_unseen.append(features)
    X_unseen = np.array(X_unseen)
    scores = ranker.predict(X_unseen)

    return scores

def load_document_content(doc_path):
    path = os.path.dirname(__file__) + "\\" + doc_path
    with open(path, 'r', encoding='utf-8') as file:
        return file.read().split()

def prepare_docs(SERP):
    docs = []
    for score, doc_path in SERP:
        doc_content = load_document_content(doc_path)
        docs.append((doc_path, ' '.join(doc_content)))  # Simpan path lengkap
    return docs


def rerank_search_results(search_query, top_k=100):
    # Jalur file untuk menyimpan model dan ranker
    lsi_model_file = 'search/letor/lsi_model.pkl'
    ranker_file = 'search/letor/lgb_ranker_lsi_model.pkl'

    # Make sure /letor ada
    if not os.path.exists('search/letor/'):
        os.makedirs('search/letor/')

    BSBI_instance = BSBIIndex(data_dir=os.path.dirname(__file__) + "/collections",
                              postings_encoding=VBEPostings,
                              output_dir=os.path.dirname(__file__) + "/index")
    BSBI_instance.load()

    SERP = BSBI_instance.retrieve_bm25(search_query, k=top_k)

    if SERP == []:
        return []

    # Periksa apakah model sudah ada
    if not os.path.isfile(lsi_model_file) or not os.path.isfile(ranker_file):

        # Load dan siapkan data
        documents = load_documents('search/qrels-folder/train_docs.txt')
        queries = load_queries('search/qrels-folder/train_queries.txt')
        q_docs_rel = load_relevance_judgments('search/qrels-folder/train_qrels.txt', queries, documents)
        dataset, group_qid_count = create_dataset(queries, documents, q_docs_rel)

        # Bangun model LSI dan latih LambdaMART
        NUM_LATENT_TOPICS = 200
        lsi_model, dictionary = create_lsi_model(documents, NUM_LATENT_TOPICS)
        X, Y = create_feature_vectors(dataset, lsi_model, dictionary)

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1
        )

        ranker.fit(X, Y, group=group_qid_count)

        # Simpan model dan ranker ke file pickle
        with open(lsi_model_file, 'wb') as f:
            pickle.dump((lsi_model, dictionary), f)

        with open(ranker_file, 'wb') as f:
            pickle.dump(ranker, f)
    else:
        # Muat model dan ranker dari file pickle
        with open(lsi_model_file, 'rb') as f:
            lsi_model, dictionary = pickle.load(f)

        with open(ranker_file, 'rb') as f:
            ranker = pickle.load(f)

    # Prediksi dan re-ranking
    docs = prepare_docs(SERP)
    scores = predict_ranking(search_query, docs, ranker, lsi_model, dictionary)

    reranked_SERP = sorted(zip(scores, [doc_path for doc_path, _ in docs]), key=lambda x: x[0], reverse=True)

    # Kembalikan hasil re-ranking
    return reranked_SERP

# Fungsi ini akan dijalankan ketika file diimpor
if __name__ == "__main__":
    search_query = "Terletak sangat dekat dengan khatulistiwa"
    reranked_results = rerank_search_results(search_query, 10)

    print("Query: ", search_query)
    BSBI_instance = BSBIIndex(data_dir='search/collections',
                              postings_encoding=VBEPostings,
                              output_dir='search/index')
    SERP = BSBI_instance.retrieve_tfidf(search_query, k=10)

    print("\nSERP:")
    for score, doc_id in SERP:
        print(f"{doc_id}: {score}")

    print("\nReranked SERP:")
    for score, doc_id in reranked_results:
        print(f"{doc_id}: {score}")
