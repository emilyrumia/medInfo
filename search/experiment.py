import os
import math
from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import Letor
from collections import defaultdict
from tqdm import tqdm


# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # TODO

    # Sumber = https://en.wikipedia.org/wiki/Discounted_cumulative_gain 
    #          PPT Pak Alfan 

    score = 0.0

    if len(ranking) == 0:
        return 0
    
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i + 1)
    return score

def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # TODO

    # Sumber = PPT Pak Alfan

    score = 0.0

    if len(ranking) == 0:
        return 0
    
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / k
    return score
    


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # TODO

    # Sumber = PPT Pak Alfan

    score = 0.0
    r = sum(ranking)

    if r == 0:
        return 0
    
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += (prec(ranking[:i], len(ranking[:i])) / r) * ranking[pos]
    return score

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    # diubah menggunakan default dict sesuai perintah TP 3
    qrels_sparse = defaultdict(lambda: defaultdict(lambda: 0)) 

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse

# >>>>> EVALUASI !

output_file = "evaluasi_no_val.txt"

# Jika dengan validation set
# 1. Buka file letor.py
# 2. Set paramater use_validation_set pada method train menjadi True
# 3. Comment kode baris 171 dan 198
# 4. Uncomment kode baris 178 dan 199
# output_file = "evaluasi_with_val.txt"

def eval_retrieval(qrels, query_file="qrels-folder/test_queries.txt", k=100, output_file=output_file):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-100 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    
    BSBI_instance.load()

    # Persiapan Letor
    letor = Letor()

    with open(query_file) as file, open(output_file, 'w') as output:

        output.write("TP 3 - Emily Rumia Naomi 2106652700\n")
        output.write("Letor tanpa validation set\n\n")
        # output.write("Letor dengan validation set\n\n")

        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_tfidf_letor = []
        dcg_scores_tfidf_letor = []
        ap_scores_tfidf_letor = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []

        rbp_scores_bm25_letor = []
        dcg_scores_bm25_letor = []
        ap_scores_bm25_letor = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            output.write("Query: " + query + "\n\n")

            """
            Evaluasi TF-IDF
            """
            docs_tfidf = []
            output.write("SERP/Ranking TF-IDF: \n")
            ranking_tfidf = []
            for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
                output.write(f"{doc:30} {score:>.3f}\n")
                with open(doc, encoding="utf-8") as file:
                    for line in file:
                        docs_tfidf.append((doc, line))
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi LETOR dari TF-IDF
            """
            output.write("\nSERP/Ranking Letor TF-IDF: \n")
            ranking_tfidf_letor = []
            sorted_did_scores = letor.predict_rank(query, docs_tfidf)
            for (doc, score) in sorted_did_scores:
                output.write(f"{doc:30} {score:>.3f}\n")
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_tfidf_letor.append(1)
                else:
                    ranking_tfidf_letor.append(0)
            rbp_scores_tfidf_letor.append(rbp(ranking_tfidf_letor))
            dcg_scores_tfidf_letor.append(dcg(ranking_tfidf_letor))
            ap_scores_tfidf_letor.append(ap(ranking_tfidf_letor))

            """
            Evaluasi BM25
            """
            docs_bm25 = []
            output.write("\nSERP/Ranking BM25: \n")
            ranking_bm25 = []
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
                output.write(f"{doc:30} {score:>.3f}\n")
                with open(doc, encoding="utf-8") as file:
                    for line in file:
                        docs_bm25.append((doc, line))
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            """
            Evaluasi LETOR dari BM25
            """
            output.write("\nSERP/Ranking Letor BM25: \n")
            ranking_bm25_letor = []
            sorted_did_scores = letor.predict_rank(query, docs_bm25)
            for (doc, score) in sorted_did_scores:
                output.write(f"{doc:30} {score:>.3f}\n")
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25_letor.append(1)
                else:
                    ranking_bm25_letor.append(0)
            rbp_scores_bm25_letor.append(rbp(ranking_bm25_letor))
            dcg_scores_bm25_letor.append(dcg(ranking_bm25_letor))
            ap_scores_bm25_letor.append(ap(ranking_bm25_letor))

            output.write("\n-------------------------------------\n\n")

        output.write("Hasil Evaluasi dengan Metrik: \n\n")
        # Untuk metrics, scorenya cukup 2 angka di belakang koma

        output.write("--- Hasil evaluasi TF-IDF ---\n")
        output.write("RBP score = {:.2f}\n".format(sum(rbp_scores_tfidf) / len(rbp_scores_tfidf)))
        output.write("DCG score = {:.2f}\n".format(sum(dcg_scores_tfidf) / len(dcg_scores_tfidf)))
        output.write("AP score  = {:.2f}\n\n".format(sum(ap_scores_tfidf) / len(ap_scores_tfidf)))

        output.write("--- Hasil evaluasi Letor TF-IDF ---\n")
        output.write("RBP score = {:.2f}\n".format(sum(rbp_scores_tfidf_letor) / len(rbp_scores_tfidf_letor)))
        output.write("DCG score = {:.2f}\n".format(sum(dcg_scores_tfidf_letor) / len(dcg_scores_tfidf_letor)))
        output.write("AP score  = {:.2f}\n\n".format(sum(ap_scores_tfidf_letor) / len(ap_scores_tfidf_letor)))

        output.write("--- Hasil evaluasi BM25 ---\n")
        output.write("RBP score = {:.2f}\n".format(sum(rbp_scores_bm25) / len(rbp_scores_bm25)))
        output.write("DCG score = {:.2f}\n".format(sum(dcg_scores_bm25) / len(dcg_scores_bm25)))
        output.write("AP score  = {:.2f}\n\n".format(sum(ap_scores_bm25) / len(ap_scores_bm25)))

        output.write("--- Hasil evaluasi Letor BM25 ---\n")
        output.write("RBP score = {:.2f}\n".format(sum(rbp_scores_bm25_letor) / len(rbp_scores_bm25_letor)))
        output.write("DCG score = {:.2f}\n".format(sum(dcg_scores_bm25_letor) / len(dcg_scores_bm25_letor)))
        output.write("AP score  = {:.2f}\n".format(sum(ap_scores_bm25_letor) / len(ap_scores_bm25_letor)))

if __name__ == '__main__':
    qrels = load_qrels()
    eval_retrieval(qrels)
