from django.shortcuts import render
import sys
import os
import re
import pickle

# from .method import hasil
from .bsbi import BSBIIndex
from .letor import Letor
from .compression import VBEPostings
from datetime import datetime


def index(request):

    start_time = datetime.now()
    totalTime=None
    sumDocs = None

    query = request.GET.get('search-bar')

    if query == None or query == "":

        context = {
            'query': query,
            'flag': 0
        }
        return render(request, 'search/index.html', context)
    
    else:

        sys.path.append(os.path.join(os.path.dirname(__file__)))
        BSBI_instance = BSBIIndex(data_dir=os.path.dirname(__file__) + "/collections",
                        postings_encoding=VBEPostings,
                        output_dir=os.path.dirname(__file__) + "/index")
 
        BSBI_instance.load()

        result = {}

# ini yang pake bm25 doang!!!!

        sorted_did_scores = BSBI_instance.retrieve_bm25(query, k=10)
        for (score, doc) in sorted_did_scores:
            print(f"{doc:30} {score:>.3f}")

        if len(sorted_did_scores) == 0:

            totalTime = 0
            sumDocs = 0

            context = {
                'query': query,
                'flag': 1,
                'totalTime': totalTime,
                'sumDocs':sumDocs
            }

            return render(request, 'search/index.html', context)
        
        else:

            # docs = []
            # for (score, doc) in sorted_did_scores:
            #     print(f"{doc:30} {score:>.3f}")
            #     path_doc = os.path.dirname(__file__) + doc.lstrip('.')
            #     with open(path_doc, encoding='utf-8') as file:
            #         for line in file:
            #             docs.append((doc,line))

            # print(docs)

            # # Persiapan Letor
            # current_filename = os.path.dirname(__file__) +'/letor/model.pkl'
            # with open(current_filename, 'rb') as f:
            #     model = pickle.load(f)
            # current_filename = os.path.dirname(__file__) +'/letor/ranker.pkl'
            # with open(current_filename, 'rb') as f:
            #     ranker = pickle.load(f)

            # letor = Letor()
            # letor.predict_rank(query, docs, model, ranker)


            for score, doc in sorted_did_scores:
                path_doc = os.path.dirname(__file__) + doc.lstrip('.')
                with open(path_doc, encoding='utf-8') as file:
                    for line in file:
                        parts = doc.split('/')
                        collection_info = f"Collections {parts[2]}: {parts[3]}"
                        result[collection_info] = line

            end_time = datetime.now()
            totalTime = end_time-start_time

            sumDocs = len(result)

            context = {
                'query': query,
                'flag': 2,
                'result' : result,
                'totalTime':totalTime,
                'sumDocs':sumDocs
            }

            return render(request, 'search/index.html', context)


def isi(request, doc):

    title = doc
    content = None
    digits = re.findall(r'\d+', doc)

    url_path = os.path.dirname(__file__) + '/collections/' + '/'.join(digits) + '.txt'
    with open(url_path, encoding='utf-8') as file:
        for line in file:
            content = line

    title1 = title.split(":")[1]
    title2 = title.split(":")[0]

    context = {
        'title': title,
        'title1': title1,
        'title2':title2,
        'content': content,
    }
   
    return render(request, 'search/isi.html', context)