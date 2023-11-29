import lightgbm as lgb
import numpy as np
import random

from scipy.spatial.distance import cosine
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from collections import defaultdict
import pickle
import os 

# Sumber = tutorial_ltr_lambdamart_lightgbm.ipynb 
# (https://colab.research.google.com/drive/1zsmcwN5fNBrVQzvE1YPEn8gJQIHXL8wa?usp=sharing)

class Letor :

    NUM_LATENT_TOPICS = 200
    NUM_NEGATIVES = 1

    # disarankan untuk menggunakan default dict 
    documents = defaultdict(lambda: defaultdict(lambda: 0)) 
    queries = defaultdict(lambda: defaultdict(lambda: 0)) 
    q_docs_rel = defaultdict(lambda: defaultdict(lambda: 0))

    group_qid_count = []
    dataset = []
    model = None
    dictionary = Dictionary()
    ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)
    
    # Variabel untuk data validasi
    val_queries = defaultdict(lambda: defaultdict(lambda: 0))
    val_q_docs_rel = defaultdict(lambda: defaultdict(lambda: 0))
    val_group_qid_count = []
    val_dataset = []
    
    def __init__(self):

        # Steps LETOR:

        # 1. Persiapkan data yang akan dilakukan re-ranking
        self.process_document()
        self.process_query()
        self.grouping_by_q_id()
        self.grouping_qid_count()

        # Validation set
        self.process_val_query()
        self.grouping_val_by_q_id()
        self.grouping_val_qid_count()

        # 2. Membuat LSI/LSA Model
        self.build_model_lsi()

        # 3. Train LightGBM LambdaMART Model
        self.train()

        # 4. Melakukan Prediksi (dipanggil saat dibutuhkan)

    def process_document(self):
        """
        Pembacaan dan pengolahan file teks untuk mengumpulkan informasi dokumen
        """

        self.documents = defaultdict(lambda: defaultdict(lambda: 0)) 

        with open("qrels-folder/train_docs.txt") as file:
            for line in file:
                values = line.split()
                if values:  # jika list tidak empty
                    doc_id = values[0]
                    content = values[1:]
                    self.documents[doc_id] = content

        print("done process document")

    def process_query(self):
        """
        Pembacaan dan pengolahan file teks untuk mengumpulkan informasi query
        """

        self.queries = defaultdict(lambda: defaultdict(lambda: 0)) 

        with open("qrels-folder/train_queries.txt",encoding="utf-8") as file:
            for line in file:
                values = line.split()
                if values:  # jika list tidak empty
                    q_id = values[0]
                    content = values[1:]
                self.queries[q_id] = content

        print("done process query")

    def grouping_by_q_id(self):
        """
        Setiap query ID memiliki daftar pasangan document ID dan relevansi.
        """

        self.q_docs_rel = defaultdict(lambda: defaultdict(lambda: 0))  

        with open("qrels-folder/train_qrels.txt") as file:
            for line in file:
                values = line.split()
                if values:  # jika list tidak empty
                    q_id = values[0]
                    doc_id = values[1]
                    rel = values[2]
                    if (q_id in self.queries) and (doc_id in self.documents):
                        if q_id not in self.q_docs_rel:
                            self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))
    
        print("done grouping by qid")

    def grouping_qid_count(self):
        """
        Jumlah dokumen yang relevan dan dokumen negatif ditambahkan ke list self.group_qid_count, 
        sementara setiap pasangan (query_text, document_text, relevance) ditambahkan ke list self.dataset. 
        Dokumen negatif diintegrasikan secara acak dengan memilih satu dokumen dari seluruh koleksi dokumen yang tersedia. 
        """

        self.group_qid_count = []
        self.dataset = []

        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
   
        print("done grouping qid count")

    def build_model_lsi(self):
        """
        Membentuk bag-of-words corpus, dan kemudian Latent Semantic Indexing/Analysis (LSI/A) model.
        """

        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
  
        print("done build model lsi")

    def vector_rep(self, text):
        """
        Menghasilkan representasi vektor untuk teks yang diberikan menggunakan sebuah model topik.
        """

        self.dictionary = Dictionary()

        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc):
        """
        Concat(vector(query), vector(document)) + informasi lain
        informasi lain -> cosine distance & jaccard similarity antara query & doc.
        """

        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def get_dataset(self):
        """
        Mengubah dataset menjadi terpisah X dan Y dimana X adalah representasi gabungan query+document,
        dan Y adalah label relevance untuk query dan document tersebut.
        """

        X = []
        Y = []
        
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def train(self, use_validation_set=False):
        """
        Melatih model ranking dengan menggunakan dataset yang telah dipersiapkan.
        """

        X_train, Y_train = self.get_dataset()

        # Latih model dengan menggunakan validation set
        if use_validation_set:
            X_val, Y_val = self.get_val_dataset()
            group_val = self.val_group_qid_count

            self.ranker.fit(X_train, Y_train, group=self.group_qid_count,
                            eval_set=[(X_val, Y_val)],
                            eval_group=[group_val], 
                            verbose=True)
            
            print("done training model with validation set")

        # Latih model tanpa menggunakan validation set
        else:
            self.ranker.fit(X_train, Y_train, group=self.group_qid_count)

            print("done training model")

    def predict_rank(self, query, docs):
        """
        Predict Ranking for Unseen Q-D Pairs.
        """

        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)

        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

        return sorted_did_scores
   
  # Keperluan Validation Set

    def process_val_query(self):
        """
        Pembacaan dan pengolahan file teks untuk mengumpulkan informasi query validation
        """

        self.val_queries = defaultdict(lambda: defaultdict(lambda: 0)) 

        with open("qrels-folder/val_queries.txt", encoding="utf-8") as file:
            for line in file:
                values = line.split()
                if values:  # jika list tidak empty
                    q_id = values[0]
                    content = values[1:]
                    self.val_queries[q_id] = content

        print("done process validation query")

    def grouping_val_by_q_id(self):
        """
        Setiap query ID memiliki daftar pasangan document ID dan relevansi untuk data validasi
        """

        self.val_q_docs_rel = defaultdict(lambda: defaultdict(lambda: 0))  

        with open("qrels-folder/val_qrels.txt") as file:
            for line in file:
                values = line.split()
                if values:  # jika list tidak empty
                    q_id = values[0]
                    doc_id = values[1]
                    rel = values[2]
                    if (q_id in self.val_queries) and (doc_id in self.documents):
                        if q_id not in self.val_q_docs_rel:
                            self.val_q_docs_rel[q_id] = []
                        self.val_q_docs_rel[q_id].append((doc_id, int(rel)))
    
        print("done grouping validation by qid")

    def grouping_val_qid_count(self):
        """
        Jumlah dokumen yang relevan dan dokumen negatif diintegrasikan secara acak dengan memilih satu dokumen
        dari seluruh koleksi dokumen yang tersedia untuk data validasi.
        """

        self.val_group_qid_count = []

        for q_id in self.val_q_docs_rel:
            docs_rels = self.val_q_docs_rel[q_id]
            self.val_group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels: 
                self.val_dataset.append((self.val_queries[q_id], self.documents[doc_id], rel))
            self.val_dataset.append((self.val_queries[q_id], random.choice(list(self.documents.values())), 0))

        print("done grouping validation qid count")
    
    def get_val_dataset(self):
        """
        Mengubah data validasi menjadi terpisah X dan Y dimana X adalah representasi gabungan query+document,
        dan Y adalah label relevance untuk query dan document tersebut.
        """

        X = []
        Y = []

        for (query, doc, rel) in self.val_dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def save_model(model, modelName):
        current_filename = os.path.dirname(__file__) +'/result/'+modelName+'.pkl'
        with open(current_filename, 'wb') as f:
            pickle.dump([model], f)

    def load_model(modelName):
        current_filename = os.path.dirname(__file__) +'/result/'+modelName+'.pkl'
        with open(current_filename, 'rb') as f:
            return pickle.load(f)

    def save_lgb_ranker(lgb_ranker, modelName):
        current_filename = os.path.dirname(__file__) +'/result/lgb_ranker_'+modelName+'.pkl'
        with open(current_filename, 'wb') as f:
            pickle.dump([lgb_ranker], f)

    def load_lgb_ranker(modelName):
        current_filename = os.path.dirname(__file__) +'/result/lgb_ranker_'+modelName+'.pkl'
        with open(current_filename, 'rb') as f:
            return pickle.load(f)