import os
from pathlib import Path
import json
import io
import pickle
from bs4 import BeautifulSoup
import re
from collections import Counter, defaultdict
from nltk import stem
from urllib.parse import urldefrag
import sys
from math import log10
import time

file_name = "ANALYST"
ps = PorterStemmer()
PARTIAL_INDEX_FILE = None
INDEX_FILE_INDEX = 0
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
    try:
        text = get_file_contents(file_path)
        tokens = tokenize(text)
        for i, token in enumerate(tokens):
            posting = (file_path.name, i + 1)  # document name/id and position
            index[token].append(posting)
        if sys.getsizeof(index) >= BATCH_SIZE:
            writePartialIndexToFile(index)
        return index
    except:
        print(f"Skipping file: {file_path}")
        return index


def writePartialIndexToFile(index: int):
    global PARTIAL_INDEX_FILE, INDEX_FILE_INDEX
    if PARTIAL_INDEX_FILE is None:
        PARTIAL_INDEX_FILE = open(f"partial_index_{INDEX_FILE_INDEX}.pickle", 'wb')
    pickle.dump(index, PARTIAL_INDEX_FILE)
    if INDEX_FILE_INDEX % 1000 == 0:
        PARTIAL_INDEX_FILE.flush()
    INDEX_FILE_INDEX += 1
    if sys.getsizeof(PARTIAL_INDEX_FILE) >= BATCH_SIZE:
        PARTIAL_INDEX_FILE.close()
        PARTIAL_INDEX_FILE = None
    return defaultdict(list)

writePartialIndexToFile.index_files = []
writePartialIndexToFile.counter = 0


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
        writePartialIndexToFile(index)
    merged_index = merge_indexes(writePartialIndexToFile.index_files)
    with open('inverted_index.pickle', 'wb') as file:
        pickle.dump(merged_index, file)
    return merged_index


def get_size(file_path):
    return os.stat(file_path).st_size / 1024  # convert to KB


def get_index_stats(index):
    with open('inverted_index.pickle', 'rb') as file:
        index = pickle.load(file)
    num_docs = 0
    for root, dirs, files in os.walk(file_name + "/"):
        num_docs += len(files)
    num_tokens = len(index.keys())
    index_size = get_size('inverted_index.pickle')
    return num_docs, num_tokens, index_size


start_time = time.time()

index = build_index(file_name + "/")

num_docs, num_tokens, index_size = get_index_stats(index)


print("Number of documents:", num_docs)
print("Number of unique tokens:", num_tokens)
print("Total size of index (Bytes):", index_size)
print(f"total processing time: {time.time() - start_time} sec")


