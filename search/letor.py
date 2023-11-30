import lightgbm as lgb
import numpy as np
import random
import pickle
import os 

from scipy.spatial.distance import cosine
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from collections import defaultdict


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

    def process_document(self):
        """
        Pembacaan dan pengolahan file teks untuk mengumpulkan informasi dokumen
        """

        self.documents = defaultdict(lambda: defaultdict(lambda: 0)) 

        with open(os.path.dirname(__file__) +"/nfcorpus/train.docs") as file:
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

        with open(os.path.dirname(__file__) +"/nfcorpus/train.vid-desc.queries", encoding='utf-8') as file:
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

        with open(os.path.dirname(__file__) +"/nfcorpus/train.3-2-1.qrel") as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
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

    def vector_rep(self, text, model):
        """
        Menghasilkan representasi vektor untuk teks yang diberikan menggunakan sebuah model topik.
        """

        self.dictionary = Dictionary()

        print("mau test")
        print(model[self.dictionary.doc2bow(text)])

        rep = [topic_value for (_, topic_value) in model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc, model):
        """
        Concat(vector(query), vector(document)) + informasi lain
        informasi lain -> cosine distance & jaccard similarity antara query & doc.
        """

        v_q = self.vector_rep(query, model)
        v_d = self.vector_rep(doc, model)
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
            X.append(self.features(query, doc, self.model))
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

    def predict_rank(self, query, docs, model, ranker):
        """
        Predict Ranking for Unseen Q-D Pairs.
        """

        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split(), model))

        X_unseen = np.array(X_unseen)

        scores = ranker.predict(X_unseen)

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
            X.append(self.features(query, doc, self.model))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y
    
    # Keperluan model dan ranker

    def save_model(self):
        current_filename = os.path.dirname(__file__) +'/letor/model.pkl'
        with open(current_filename, 'wb') as f:
            pickle.dump([self.model], f)

    def load_model(self):
        # current_filename = os.path.dirname(__file__) +'/letor/model.pkl'
        current_filename = os.path.dirname(__file__) +'/letor/lsi_model.pkl'
        with open(current_filename, 'rb') as f:
            return pickle.load(f)

    def save_ranker(self):
        current_filename = os.path.dirname(__file__) +'/letor/ranker.pkl'
        with open(current_filename, 'wb') as f:
            pickle.dump([self.ranker], f)

    def load_ranker(self):
        # current_filename = os.path.dirname(__file__) +'/letor/ranker.pkl'
        current_filename = os.path.dirname(__file__) +'/letor/lgb_ranker_lsi_model.pkl'
        with open(current_filename, 'rb') as f:
            return pickle.load(f)
        
if __name__ == "__main__":
   
    # Steps LETOR:

    letor = Letor()

    # 1. Persiapkan data yang akan dilakukan re-ranking
    letor.process_document()
    letor.process_query()
    letor.grouping_by_q_id()
    letor.grouping_qid_count()

    # Validation set
    # letor.process_val_query()
    # letor.grouping_val_by_q_id()
    # letor.grouping_val_qid_count()

    # 2. Membuat LSI/LSA Model
    letor.build_model_lsi()
    letor.save_model()

    # 3. Train LightGBM LambdaMART Model
    letor.train()
    letor.save_ranker()

    # 4. Melakukan Prediksi (dipanggil saat dibutuhkan)