import csv
import re
import math
import time
import tkinter as tk
from tkinter import messagebox
from nltk import stem
from scipy.spatial.distance import cosine
import numpy as np
from collections import defaultdict


STEMMER = stem.PorterStemmer()

INDEX_FILE = 'index/index.csv'
ID_TO_URL_FILE = 'idToUrl.csv'
CACHE_SIZE_LIMIT = 1000
# Increase the field size limit
csv.field_size_limit(1024 * 1024 * 1024) 


# Cache to store a portion of the inverted index
cache = defaultdict(list)


def retrieve_index(word):
    if word in cache:
        return word, cache[word]

    if any(char.isdigit() for char in word):
        # Word contains a digit, search in index.csv file
        with open(INDEX_FILE, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                token = row[0]
                if token == word:
                    postings = [tuple(posting.split(':')) for posting in row[1].split(',')]

                    if len(cache) >= CACHE_SIZE_LIMIT:
                        cache.pop(next(iter(cache)))

                    cache[token] = postings
                    return token, postings
    else:
        # Word starts with an alphabet, determine the file name based on the first letter
        file_name = f"index/index_{word[0].lower()}.csv"
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                token = row[0]
                if token == word:
                    postings = [tuple(posting.split(':')) for posting in row[1].split(',')]

                    if len(cache) >= CACHE_SIZE_LIMIT:
                        cache.pop(next(iter(cache)))

                    cache[token] = postings
                    return token, postings

    return word, []




def load_id_to_url():
    id_to_url = {}
    with open(ID_TO_URL_FILE, 'r', newline='') as id_file:
        reader = csv.reader(id_file)
        next(reader)  # Skip the header row
        for row in reader:
            doc_id = int(row[0])  # Convert the key to an integer
            url = row[1]
            id_to_url[doc_id] = url
    return id_to_url


def process_query(query, important_words="important_words.txt"):
    query_tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).split()]
    query_tokens, query_vector = mod_query_vector(query_tokens)
    relevant_indexes = {key: value for key, value in [retrieve_index(word) for word in query_tokens] if value}
    if not relevant_indexes:
        return []  # Return empty list if no relevant indexes found
    
    # Convert important_words to a set for faster membership testing
    with open(important_words, 'r') as file:
        important_words_set = set(file.read().splitlines())
    
    vectors = create_doc_tfidf_matrix(query_tokens, relevant_indexes)
    vectors, avg_max = get_best_quartile(vectors)
    query_vector = normalize(query_vector)
    normed = {document: normalize(vectors[document]) for document in vectors}
    cosine_rank = cosine_ranking(query_vector, normed)
    rankings = {}
    for doc in cosine_rank:
        if len(query_tokens) < 3:
            rankings[doc] = np.sum(vectors[doc], dtype=float)
        else:
            rankings[doc] = cosine_rank[doc] * 0.6 + 0.4 * np.mean(vectors[doc], dtype=float) / avg_max
            score = cosine_rank[doc] * 0.6 + 0.4 * np.mean(vectors[doc], dtype=float) / avg_max
            
            # Use set operations for membership testing
            important_word_score = sum(1 for word in query_tokens if word in important_words_set)
            score *= (1 + important_word_score)
            
            rankings[doc] = score
    best = sorted(rankings, key=lambda x: -rankings[x])
    top_urls = [id_to_url[int(doc)] for doc in best]
    return top_urls[:min(len(top_urls), 5)]



def cosine_ranking(query_vector: dict, vector: dict):
    return {document: np.nansum(query_vector * vector[document]) for document in vector}


def normalize(vector):
    vector = np.array(vector, dtype=float)
    length = np.nansum(vector ** 2) ** 0.5
    vector = vector / length
    return vector


def get_best_quartile(vector):
    sum_vector = {}
    for doc in vector:
        try:
            mean_value = np.mean(np.array(vector[doc], dtype=float))
        except KeyError:
            mean_value = 0
        sum_vector[doc] = mean_value

    best = sorted(sum_vector, key=lambda x: -sum_vector[x])
    extract = math.floor(len(sum_vector) / 4) if math.floor(len(sum_vector) / 4) >= 10 else len(sum_vector)
    if extract > 500:
        extract = 500
    best = best[0:extract + 1]
    return {doc: vector[doc] for doc in best}, np.mean(np.array(vector[doc], dtype=float))




def create_doc_tfidf_matrix(terms: list, inverted_index: dict) -> dict:
    vector = {}  # dictionary - documents are keys, tf-idf expressed as a list initially
    for i in range(len(terms)):
        term = terms[i]
        if term in inverted_index:
            for document, tfidf_scores in inverted_index[term]:
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
        if term in query:
            q_vect[query_set_list.index(term)] += 1
        else:
            q_vect[query_set_list.index(term)] = 1
    return query_set_list, np.array(q_vect)


def perform_search():
    query = entry.get()
    if query:
        t1 = time.time()
        similar_doc_list = process_query(query)
        result_text.delete(1.0, tk.END)
        if len(similar_doc_list) == 0:
            result_text.insert(tk.END, "No result found")
        else:
            for i in range(min(5, len(similar_doc_list))):
                result_text.insert(tk.END, f"{similar_doc_list[i]}\n\n")
        t2 = time.time()
        time_label.config(text=f"{t2 - t1:.2f} seconds")
    else:
        result_text.delete(1.0, tk.END)


# Load the idToUrl mapping file
id_to_url = load_id_to_url()

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
