import json
import re
import math
import time
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from nltk import stem
from scipy.spatial.distance import cosine
import numpy as np

STEMMER = stem.PorterStemmer()

def load_index():
    index_data = {}
    with open('index.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            index_data.update(data)
    return index_data


def load_idToUrl():
    # Load the idToUrl mapping file
    with open('idToUrl.json', 'r') as id_file:
        id_to_url = json.load(id_file)
    return id_to_url

def process_query(query, index_data, id_to_url):
    query_tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).split()]
    #print(query_tokens)
    query_tokens, query_vector = mod_query_vector(query_tokens)
    #print(query_tokens, query_vector)
    relevant_indexes = {key: value for key, value in [retrieve_index(word, index_data) for word in query_tokens]}
    #print(relevant_indexes)
    vectors = create_doc_tfidf_matrix(query_tokens, relevant_indexes)
    #print(vectors)
    vectors, avg_max = get_best_quartile(vectors)
    # print(vectors)
    # print(avg_max)
    query_vector = normalize(query_vector)
    #print(query_vector)
    normed = {document: normalize(vectors[document]) for document in vectors}
    #print(normed)
    cosine_rank = cosine_ranking(query_vector, normed)
    #print(cosine_rank)

    rankings = {}
    for doc in cosine_rank:
        if len(query_tokens) < 3:
            rankings[doc] = np.sum(vectors[doc])
        else:
            rankings[doc] = cosine_rank[doc] * 0.6 + 0.4 * np.mean(vectors[doc]) / avg_max
    best = sorted(rankings, key=lambda x: -rankings[x])
    #print(best)
    top_urls = [id_to_url[str(doc)] for doc in best]
    return top_urls[:min(len(top_urls), 5)]


def cosine_ranking(query_vector: dict, vector: dict):
    return {document: np.nansum(query_vector * vector[document]) for document in vector}

def normalize(vector):
    vector = np.array(vector, dtype=float)
    length = np.nansum(vector ** 2) ** 0.5
    vector = vector / length
    return vector

def get_best_quartile(vector):
    sum_vector = {doc: np.mean(np.array(vector[doc])) for doc in vector}
    best = sorted(sum_vector, key=lambda x: -sum_vector[x])
    extract = math.floor(len(sum_vector) / 4) if math.floor(len(sum_vector) / 4) >= 10 else len(sum_vector)
    if extract > 500:
        extract = 500
    best = best[0:extract + 1]
    return {doc: vector[doc] for doc in best}, np.mean(vector[best[0]])


def create_doc_tfidf_matrix(terms: list, inverted_index: dict) -> dict:
    vector = {}  # dictionary - documents are keys, tf-idf expressed as a list initially
    for i in range(len(terms)):
        for document, tfidf_scores in inverted_index[terms[i]]:
            if document in vector:
                vector[document][i] = tfidf_scores
            else:
                vector[document] = [0 for _ in terms]
                vector[document][i] = tfidf_scores
    return vector


def mod_query_vector(query: list) -> tuple:
    query_set_list = list(set(query))
    q_vect = [0 for _ in query_set_list]
    for term in query:
        if term in query: q_vect[query_set_list.index(term)] += 1
        else: q_vect[query_set_list.index(term)] = 1
    return query_set_list, np.array(q_vect)

def retrieve_index(word, index_data):
    if word in index_data:
        return word, index_data[word]
    raise ValueError("Word not found in the inverted index.")



def perform_search():
    query = entry.get()
    if query:
        t1 = time.time()
        similar_doc_list = process_query(query, index_data, idToUrl_data)
        result_text.delete(1.0, tk.END)
        for i in range(min(5, len(similar_doc_list))):
            result_text.insert(tk.END, f"{similar_doc_list[i]}\n\n")
        t2 = time.time()
        time_label.config(text=f"{t2 - t1:.2f} seconds")
    else:
        result_text.delete(1.0, tk.END)


index_data = load_index()
idToUrl_data = load_idToUrl()

# process_query("cristina lopes", index_data, idToUrl_data)

root = tk.Tk()
root.title("Search Engine")

frame = tk.Frame(root)
frame.pack(pady=10)

label = tk.Label(frame, text="Enter your query:")
label.pack(side=tk.LEFT)

entry = tk.Entry(frame, width=40)
entry.pack(side=tk.LEFT)

search_button = tk.Button(frame, text="Search", command=perform_search)
search_button.pack(side=tk.LEFT, padx=10)

result_text = tk.Text(root, height=30, width=100)
result_text.pack(pady=10, padx=10)

time_label = tk.Label(root, text="")
time_label.pack()

root.mainloop()

