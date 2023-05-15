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

STEMMER = stem.PorterStemmer()
PATH = "ANALYST/"
partialIndexPath = "partial_index_"
fullIndexPath = "index.jsonl"
importantTags = ["h1", "h2", "h3", "strong", "b"]
batch_size = 1024 * 1024 * 1024


class Index:
    def __init__(self, path) -> None:
        self.filepath = path
        self.idToUrl = {}
        self.urlToId = {}
        self.inverted_index = defaultdict(list)
        self.batchCounter = 0
        self.numberOfFilesProcessed = 0
        self.document_number = 0
        self.numberOfTokensProcessed = 0
        self.partialIndexFiles = []
        self.partial_file_index = 0

    def generate_line(self, filename, chunk_size=1024 * 1024):
        """This function reads file in chunks (1GB). It returns a single key and posting pair."""
        # Used BufferReader because this allows us to read the data in chunks without reading line by line and could be efficient for very large files
        with open(filename, "rb") as file:
            buffer = io.BufferedReader(file, chunk_size)
            while True:
                chunk = buffer.readline()
                if not chunk:
                    break
                line = chunk.decode().strip()
                yield json.loads(line)

    def write_to_pickle(self):
        """Writing the data to pickle file."""
        # We have used pickle here because writing to pickle file would be much more efficient than writing to json file.
        with open("idToUrl.pickle", "wb") as f:
            buffer = io.BufferedWriter(f)
            pickle.dump(self.idToUrl, buffer)
            buffer.flush()

    def tfidf_score(self, postingsList: list):
    frequency_tfidf = list()
    for post in postingsList:
        tfidf = round((1 + log10(post[1])) * (log10(self.filesProcessed / len(postingsList))), 2)
        frequency_tfidf.append((post[0], tfidf))
    return frequency_tfidf

    def build_index(self):
        path = Path(self.filepath)
        for directory in path.iterdir():
            # print(directory)
            if not directory.is_dir():
                continue
            for file in directory.iterdir():
                # print(file)
                if not file.is_file():
                    continue

                self.process_file(file)
                self.document_number += 1

                if sys.getsizeof(self.inverted_index) >= batch_size:
                    self.partial_index()

        if self.inverted_index:
            self.partial_index()

    def process_file(self, file):
        try:
            with open(file) as jsonFile:
                file_dict = json.load(jsonFile)
                url = urldefrag(file_dict["url"]).url

                self.urlToId[url] = self.document_number
                self.idToUrl[self.document_number] = url

                raw_content = file_dict["content"]
                soup = BeautifulSoup(raw_content, "html.parser")
                important_words = self.extract_important_words(soup)
                token_frequency = self.tokenize_content(soup)

                for token, frequency in token_frequency.items():
                    if token not in self.inverted_index:
                        self.inverted_index[token] = []

                    if token in important_words:
                        self.inverted_index[token].append((self.document_number, frequency * 100))
                    else:
                        self.inverted_index[token].append((self.document_number, frequency))
                self.numberOfFilesProcessed += 1
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return None

        return file

    def extract_important_words(self, content):
        important_words = set()
        for tags in content.find_all(importantTags):
            for words in re.sub(r"[^a-zA-Z0-9\s]", "", tags.text.lower()).split():
                important_words.add(STEMMER.stem(words))
        return important_words

    def tokenize_content(self, content):
        content = content.find_all()
        frequency = Counter()
        for line in content:
            if not line.text:
                continue
            tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", line.text.lower()).split()]
            frequency.update(tokens)
        return frequency

    def partial_index(self):
        with open(f"partial_index_{self.partial_file_index}.pickle", "wb") as partialFile:
            sorted_index = sorted(self.inverted_index.items(), key=lambda x: x[0])
            self.inverted_index = {}
            for key, value in sorted_index:
                keyVal = pickle.dumps({key: value})
                partialFile.write(keyVal + "\n")
            self.partial_file_index += 1
            self.partialIndexFiles.append(f"partial_index_{self.partial_file_index}.pickle")

    def merge_index(self):
        pass

    def printStats(self):
        # totalSize = os.stat("index.jsonl").st_size
        print("Number of files processed:", self.numberOfFilesProcessed)
        print("Number of unique tokens:", self.numberOfTokensProcessed)
        print("Total disk size (bytes):")


if __name__ == "__main__":
    startTime = time.strftime("%H:%M:%S")
    print(startTime)
    index = Index(PATH)
    index.build_index()
    index.write_to_pickle()
    index.printStats()
    endTime = time.strftime("%H:%M:%S")
    print(endTime)
