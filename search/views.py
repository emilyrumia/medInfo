import os
import re
from datetime import datetime

from django.shortcuts import render

from .letor import rerank_search_results


def index(request):
    start_time = datetime.now()
    totalTime = None
    sumDocs = None
    num_results = request.GET.get('num_results', 10)  # Default adalah 10

    query = request.GET.get('search-bar')

    if query == None or query == "":

        context = {
            'query': query,
            'flag': 0
        }
        return render(request, 'search/index.html', context)

    else:

        result = {}

        # ini yang pake bm25 doang!!!!

        sorted_did_scores = rerank_search_results(query, int(num_results))
        for (score, doc) in sorted_did_scores:
            print(f"{doc:30} {score:>.3f}")

        if len(sorted_did_scores) == 0:  # jika tidak ada hasil

            totalTime = 0
            sumDocs = 0

            context = {
                'query': query,
                'flag': 1,
                'totalTime': totalTime,
                'sumDocs': sumDocs
            }

            return render(request, 'search/index.html', context)

        else:  # jika ada hasil

            for score, doc in sorted_did_scores:
                path_doc = os.path.dirname(__file__) + "\\" + doc.lstrip('.')
                with open(path_doc, encoding='utf-8') as file:
                    for line in file:
                        parts = doc.split('/')
                        collection_info = f"Collections {parts[2]}: {parts[3]}"
                        result[collection_info] = line

            end_time = datetime.now()
            totalTime = end_time - start_time

            sumDocs = len(result)

            context = {
                'query': query,
                'flag': 2,
                'result': result,
                'totalTime': totalTime,
                'sumDocs': sumDocs
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
        'title2': title2,
        'content': content,
    }

    return render(request, 'search/isi.html', context)
