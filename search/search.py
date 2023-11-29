from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import os

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
BSBI_instance.load()

# Persiapan Letor
letor = Letor()
model = letor.load_model()[0]
ranker = letor.load_ranker()[0]

queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri"
           ,"Terletak sangat dekat dengan khatulistiwa"]

for query in queries:
    print("\nQuery  : ", query)
    print("\nTF-IDF")
    print("SERP/Ranking: ")
    docs = []
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
        print(f"{doc:30} {score:>.3f}")
        with open(doc, encoding='utf-8') as file:
            for line in file:
                docs.append((doc, line))
    print("\nLETOR")
    sorted_did_scores  = letor.predict_rank(query, docs, model, ranker)
    print("SERP/Ranking :")
    for (doc, score) in sorted_did_scores:
        print(f"{doc:30} {score:>.3f}")
    print("\n-------------------------------------\n\n")