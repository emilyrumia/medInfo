from django.shortcuts import render
import sys
import os
from .method import hasil
from datetime import datetime


def index(request):
    query = request.GET.get('search-bar')
    if query == None or query == "":
        print("kosong")
        context = {
            'query': query,
            # ganti flag ini buat coba 0, 1, 2
            'flag': 2
        }
        return render(request, 'search/index.html', context)
    else:
        print(query)
        sorted_did_scores = hasil(100, query)
        if sorted_did_scores is None:
            context = {
                'query': query,
                'flag': 1
            }
            return render(request, 'home/index.html', context)
        else:
            result = {}
            for doc in result:
                data_dir = os.path.dirname(__file__) + "/collections"
                print(data_dir)
                # text = open(data_dir).read()
                # coll_ids = int((doc).split("/")[0])
                # ids = int((doc).split("/")[1].split(".")[0])
                # result[str(doc)] = [text, coll_ids, ids]
            context = {
                'query': query,
                'flag': 2
            }
            return render(request, 'home/index.html', context)


def isi(request):
    # n = int(doc_id)//100 + 1
    # dir_text = os.path.dirname(__file__) + "/collection/" + str(n) + "/" + str(doc_id) +".txt"
    # text = open(dir_text).read()

    # context = {
    #     'doc_id': doc_id,
    #     'text': text,
    # }
   
    return render(request, 'search/isi.html')