import os 
import numpy as np
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import Letor

def vector_rep(text, model):
    dictionary = Dictionary()
    NUM_LATENT_TOPICS = 200
    rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
    return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

def features(query, doc, model):
    v_q = vector_rep(query, model)
    v_d = vector_rep(doc, model)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)
    return v_q + v_d + [jaccard] + [cosine_dist]

def predict_rank(query, docs, model, ranker):
    # bentuk ke format numpy array
    X_unseen = []
    for doc_id, doc in docs:
        print(doc.split())
        X_unseen.append(features(query.split(), doc.split(), model))

    X_unseen = np.array(X_unseen)

    # hitung scores
    scores = ranker.predict(X_unseen)
    
    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    return sorted_did_scores


def hasil(k=10, query=''):
    print("test")

    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_dir=os.path.dirname(__file__) + "/collections",
                              postings_encoding=VBEPostings,
                              output_dir=os.path.dirname(__file__) + "/index")
    print("sampe sini")
    BSBI_instance.load()
    print("lewat akhirnya")

    # Persiapan Letor
    letor = Letor()
    model = letor.load_model()[0]
    ranker = letor.load_ranker()[0]

    print("test 1")

    # BM25
    docs = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
        print(score, doc)
        path_doc = os.path.dirname(__file__) + doc.lstrip('.')
        with open(path_doc, encoding='utf-8') as file:
            for line in file:
                docs.append((doc, line))
                print((doc, line))

    print("test 2")

    # Letor
    sorted_did_scores  = letor.predict_rank(query, docs, model, ranker)

    return sorted_did_scores
