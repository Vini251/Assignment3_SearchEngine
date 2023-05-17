import json
from nltk import stem
import re
import tkinter as tk
import time

STEMMER = stem.PorterStemmer()

def get_similar_docs(query):
    # Load the index file
    with open('index.jsonl', 'r') as file:
        index_data = file.readlines()

    # Load the idToUrl mapping file
    with open('idToUrl.json', 'r') as id_file:
        id_to_url = json.load(id_file)

    # Initialize a dictionary to store document numbers and their matching score
    similar_docs = {}

    # Split the query into individual tokens
    query_tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).split()]

    # Iterate over each line in the index file
    for line in index_data:
        # Parse the JSON object
        data = json.loads(line)

        # Iterate over each token and its corresponding document numbers
        for token, doc_scores in data.items():
            if token in query_tokens:
                # Iterate over each document number and its corresponding TF-IDF score
                for doc_score in doc_scores:
                    doc_number = doc_score[0]
                    tfidf_score = doc_score[1]

                    # Add the document number and score to the dictionary
                    if doc_number in similar_docs:
                        similar_docs[doc_number] += tfidf_score
                    else:
                        similar_docs[doc_number] = tfidf_score

    # Sort the similar_docs dictionary based on the cumulative TF-IDF scores
    sorted_docs = sorted(similar_docs.items(), key=lambda x: x[1], reverse=True)

    # Extract the document numbers from the sorted list
    similar_doc_numbers = [doc_number for doc_number, _ in sorted_docs]

    # Retrieve the corresponding URLs from the idToUrl mapping
    similar_doc_urls = [id_to_url.get(str(doc_number)) for doc_number in similar_doc_numbers]

    return similar_doc_urls


def search():
    query = entry.get()
    if query:
        t1 = time.time()
        similar_doc_list = get_similar_docs(query)
        result_text.delete(1.0, tk.END)
        for i in range(min(5, len(similar_doc_list))):
            result_text.insert(tk.END, f"{similar_doc_list[i]}\n\n")
        t2 = time.time()
        time_label.config(text=f"{t2 - t1:.2f} seconds")
    else:
        result_text.delete(1.0, tk.END)

root = tk.Tk()
root.title("Search Engine")

frame = tk.Frame(root)
frame.pack(pady=10)

label = tk.Label(frame, text="Enter your query:")
label.pack(side=tk.LEFT)

entry = tk.Entry(frame, width=40)
entry.pack(side=tk.LEFT)

search_button = tk.Button(frame, text="Search", command=search)
search_button.pack(side=tk.LEFT, padx=10)

result_text = tk.Text(root, height=30, width=100)
result_text.pack(pady=10, padx=10)

time_label = tk.Label(root, text="")
time_label.pack()

root.mainloop()


