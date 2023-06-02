import csv
import re
import math
import time
import tkinter as tk
from nltk import stem
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
    """
    Retrieves the postings for a given word from the index.
    If the word is found in the cache, returns the cached postings.
    Otherwise, searches for the word in the index files and returns the postings if found.
    """

    if word in cache:
        # Word is found in the cache, return the cached postings
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

    # Word not found, return the word and empty postings
    return word, []





def load_id_to_url():
    """
    Loads the mapping of document IDs to URLs from the ID_TO_URL_FILE.
    Returns a dictionary containing the mapping.
    """

    id_to_url = {}

    # Open the ID_TO_URL_FILE for reading
    with open(ID_TO_URL_FILE, 'r', newline='') as id_file:
        reader = csv.reader(id_file)
        next(reader)  # Skip the header row

        # Iterate over each row in the file
        for row in reader:
            doc_id = int(row[0])  # Convert the key to an integer
            url = row[1]
            id_to_url[doc_id] = url

    # Return the dictionary containing the mapping of document IDs to URLs
    return id_to_url



def process_query(query, important_words="important_words.txt"):
    # Tokenize the query, apply stemming, and create a query vector
    query_tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).split()]
    query_tokens, query_vector = mod_query_vector(query_tokens)

    # Retrieve relevant indexes for the query tokens
    relevant_indexes = {key: value for key, value in [retrieve_index(word) for word in query_tokens] if value}

    if not relevant_indexes:
        return []  # Return empty list if no relevant indexes found

    # Read and convert important_words to a set for faster membership testing
    with open(important_words, 'r') as file:
        important_words_set = set(file.read().splitlines())

    # Create the TF-IDF matrix for the query tokens and relevant indexes
    vectors = create_doc_tfidf_matrix(query_tokens, relevant_indexes)

    # Filter the vectors based on the best quartile and calculate the average maximum
    vectors, avg_max = get_best_quartile(vectors)

    # Normalize the query vector and create a dictionary of normalized document vectors
    query_vector = normalize(query_vector)
    normed = {document: normalize(vectors[document]) for document in vectors}

    # Calculate the cosine ranking for the query vector and normalized document vectors
    cosine_rank = cosine_ranking(query_vector, normed)

    rankings = {}

    for doc in cosine_rank:
        if len(query_tokens) < 3:
            rankings[doc] = np.sum(vectors[doc], dtype=float)
        else:
            score = cosine_rank[doc] * 0.6 + 0.4 * np.mean(vectors[doc], dtype=float) / avg_max

            # Use set operations for membership testing
            important_word_score = sum(1 for word in query_tokens if word in important_words_set)
            score *= (1 + important_word_score)

            rankings[doc] = score

    # Sort the rankings and retrieve the corresponding top URLs
    best = sorted(rankings, key=lambda x: -rankings[x])
    top_urls = [id_to_url[int(doc)] for doc in best]

    # Return the list of top URLs
    return top_urls



def cosine_ranking(query_vector: dict, vector: dict):
    """Calculates the cosine ranking between a query vector and document vectors."""
    return {document: np.nansum(query_vector * vector[document]) for document in vector}


def normalize(vector):
    """Normalizes a vector by dividing each element by its Euclidean length."""
    vector = np.array(vector, dtype=float)
    length = np.nansum(vector ** 2) ** 0.5
    vector = vector / length
    return vector


def get_best_quartile(vector):
    """Extracts the best quartile of document vectors based on mean values and calculates the average maximum."""
    sum_vector = {}

    # Calculate the mean value for each document vector
    for doc in vector:
        try:
            mean_value = np.mean(np.array(vector[doc], dtype=float))
        except KeyError:
            mean_value = 0
        sum_vector[doc] = mean_value

    # Sort the document vectors based on the mean values in descending order
    best = sorted(sum_vector, key=lambda x: -sum_vector[x])

    # Determine the number of documents to extract for the best quartile
    extract = math.floor(len(sum_vector) / 4) if math.floor(len(sum_vector) / 4) >= 10 else len(sum_vector)

    # Limit the number of documents to extract to a maximum of 500
    if extract > 500:
        extract = 500

    # Extract the document vectors of the best quartile
    best = best[0:extract + 1]

    # Return the document vectors of the best quartile and the average maximum
    return {doc: vector[doc] for doc in best}, np.mean(np.array(vector[doc], dtype=float))




def create_doc_tfidf_matrix(terms: list, inverted_index: dict) -> dict:
    """Creates a document TF-IDF matrix based on the given terms and inverted index."""
    vector = {}  # dictionary - documents are keys, tf-idf expressed as a list initially

    # Iterate over the terms in the query
    for i in range(len(terms)):
        term = terms[i]

        # Check if the term exists in the inverted index
        if term in inverted_index:

            # Iterate over the (document, tf-idf) scores for the term
            for document, tfidf_scores in inverted_index[term]:

                # Check if the document already has a vector in the TF-IDF matrix
                if document in vector:
                    vector[document][i] = tfidf_scores
                else:
                    # If the document does not have a vector, initialize it with zeros
                    vector[document] = [0 for _ in terms]
                    vector[document][i] = tfidf_scores

    return vector



def mod_query_vector(query: list) -> tuple:
    """Modifies the query vector by removing duplicates and creating a query vector representation."""
    query_set_list = list(set(query))  # Get unique terms in the query
    q_vect = [0 for _ in query_set_list]  # Initialize the query vector with zeros

    # Iterate over each term in the original query
    for term in query:
        if term in query:
            q_vect[query_set_list.index(term)] += 1  # Increment the frequency of the term in the query vector
        else:
            q_vect[query_set_list.index(term)] = 1  # Set the frequency of the term in the query vector to 1

    return query_set_list, np.array(q_vect)


def perform_search():
    """Performs a search based on the user input query and displays the search results."""
    query = entry.get()  # Retrieve the query from the user input

    if query:
        t1 = time.time()  # Measure the start time for performance measurement

        similar_doc_list = process_query(query)  # Process the query and get a list of similar documents

        result_text.delete(1.0, tk.END)  # Clear the existing contents in the result_text widget

        if len(similar_doc_list) == 0:
            result_text.insert(tk.END, "No result found")  # Display a message when no similar documents are found
        else:
            for i in range(min(5, len(similar_doc_list))):  # Iterate over a maximum of 5 similar documents
                result_text.insert(tk.END, f"{similar_doc_list[i]}\n\n")  # Display each similar document URL

            result_text.insert(tk.END, f"Total number of urls: {len(similar_doc_list)}\n\n")  # Display the total number of URLs found

        t2 = time.time()  # Measure the end time for performance measurement
        time_label.config(text=f"{t2 - t1:.2f} seconds")  # Update the time_label widget with the search time

    else:
        result_text.delete(1.0, tk.END)  # Clear the existing contents in the result_text widget if no query is provided


# Load the idToUrl mapping file
id_to_url = load_id_to_url()

#Local GUI
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
