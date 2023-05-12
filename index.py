from pathlib import Path
import sys
import pickle
import os
import re
from nltk.stem import PorterStemmer
from collections import defaultdict

ps = PorterStemmer()
BATCH_SIZE = 1024 * 1024 * 1024  # 1 GB


def tokenize(text):
    tokens = []
    for word in re.findall(r'\w+', text):
        tokens.append(ps.stem(word.lower()))
    return tokens


def get_file_contents(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        return file.read()


def index_document(file_path, index):
    text = get_file_contents(file_path)
    tokens = tokenize(text)
    for i, token in enumerate(tokens):
        posting = (file_path.name, i + 1)  # document name/id and position
        index[token].append(posting)
    if sys.getsizeof(index) >= BATCH_SIZE:
        PartialIndex(index)
    return index


def PartialIndex(index):
    index_file = f"partial_index_{PartialIndex.counter}.pickle"
    with open(index_file, 'wb') as file:
        pickle.dump(index, file)
    PartialIndex.index_files.append(index_file)
    PartialIndex.counter += 1
    return defaultdict(list)


PartialIndex.index_files = []
PartialIndex.counter = 0


def merge_indexes(index_files):
    merged_index = defaultdict(list)
    for index_file in index_files:
        with open(index_file, 'rb') as file:
            index = pickle.load(file)
        for key, postings in index.items():
            merged_index[key].extend(postings)
        os.remove(index_file)
    return merged_index


def build_index(directory):
    index = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = Path(root) / file_name
            index = index_document(file_path, index)
    if index:
        PartialIndex(index)
    merged_index = merge_indexes(PartialIndex.index_files)
    with open('inverted_index.pickle', 'wb') as file:
        pickle.dump(merged_index, file)
    return merged_index


def get_size(file_path):
    return os.stat(file_path).st_size / 1024  # convert to KB


def get_index_stats(index):
    with open('inverted_index.pickle', 'rb') as file:
        index = pickle.load(file)
    num_docs = 0
    for root, dirs, files in os.walk("DEV/"):
        num_docs += len(files)
    num_tokens = len(index.keys())
    index_size = get_size('inverted_index.pickle')
    return num_docs, num_tokens, index_size


index = build_index("DEV/")

num_docs, num_tokens, index_size = get_index_stats(index)

print("Number of documents:", num_docs)
print("Number of unique tokens:", num_tokens)
print("Total size of index (Bytes):", index_size)



# Test - Alex