import os
from pathlib import Path
import json
from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk import stem
from urllib.parse import urldefrag
import sys
from math import log10
import time

STEMMER = stem.PorterStemmer()
PATH = "DEV/"
fullIndexPath = "index.jsonl"
importantTags = ["h1", "h2", "h3", "strong", "b"]
batch_size = 3000000


class Index:
    def __init__(self, path) -> None:
        self.filepath = path
        self.idToUrl = {}
        self.urlToId = {}
        self.inverted_index = {}
        self.numberOfFilesProcessed = 0
        self.document_number = 0
        self.numberOfTokensProcessed = 0
        self.partialIndexFiles = []
        self.partial_file_index = 0
        self.important_words = set()

    def write_important_words(self):
        with open("important_words.txt", 'w') as file:
            for word in self.important_words:
                file.write(word + '\n')

    def generate_line(self, filename):
        """This function reads file. It returns a single key and posting pair."""
        # Used BufferReader because this allows us to read the data in chunks without reading line by line and could be efficient for very large files
        with open(filename) as f:
            for line in f:
                yield json.loads(line)

    def write_to_json(self):
        """Writing the data to json file."""
        with open("idToUrl.json", "w") as f:
            jsonObj = json.dumps(self.idToUrl)
            f.write(jsonObj)

    def tfidf_score(self, postingsList: list):
        frequency_tfidf = list()
        for post in postingsList:
            tfidf = round((1 + log10(post[1])) * (log10(self.numberOfFilesProcessed / len(postingsList))), 2)
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

                for line in self.generate_line(file):
                    file_dict = line
                    url = urldefrag(file_dict["url"]).url

                    if url in self.urlToId:
                        continue
                    self.urlToId[url] = self.document_number
                    self.idToUrl[self.document_number] = url

                    raw_content = file_dict["content"]
                    soup = BeautifulSoup(raw_content, features = "html.parser")
                    
                
                    for tags in soup.find_all(importantTags):
                        for words in re.sub(r"[^a-zA-Z0-9\s]", "", tags.text.lower()).split():
                            self.important_words.add(STEMMER.stem(words))
                    
                    soup = soup.find_all()
                    frequency = Counter()
                    for line in soup:
                        if not line.text:
                            continue
                        tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", line.text.lower()).split()]
                        frequency.update(tokens)

                    for token, frequency in frequency.items():
                        if token not in self.inverted_index:
                            self.inverted_index[token] = []

                        if token in self.important_words:
                            self.inverted_index[token].append((self.document_number, frequency * 100))
                        else:
                            self.inverted_index[token].append((self.document_number, frequency))
                self.numberOfFilesProcessed += 1
                self.document_number += 1
                
                if sys.getsizeof(self.inverted_index) >= batch_size:
                    self.partial_index()

        if self.inverted_index:
            self.partial_index()


    def partial_index(self):
        path = f"partial_index_{self.partial_file_index}.jsonl"
        with open(path, "w") as partialFile:
            sorted_index = sorted(self.inverted_index.items(), key=lambda x: x[0])
            self.inverted_index = {}
            for key, value in sorted_index:
                keyVal = json.dumps({key: value})
                partialFile.write(keyVal + "\n")
        self.partial_file_index += 1
        self.partialIndexFiles.append(path)

    def merge_files(self):
        """ opens main index and all partial indices, sorts and writes all partial indices into main """

        with open(fullIndexPath, 'w') as final:  # open final index
            file_generators = [self.generate_line(filename) for filename in self.partialIndexFiles]  # generator obj for each partial
            next_words = []  # stores most recent word yielded from each generator
            toBeRemoved = set()  # stores empty generators queued for removal

            for i, gen in enumerate(file_generators):
                currGensNext = list(next(gen).items())
                next_words.append((currGensNext[0][0], currGensNext[0][1], file_generators[i]))

            while file_generators:  # loop until all generators empty
                next_words.sort(key=lambda x: x[0])  # sort yielded words
                getNextVals = [next_words[0][2]]  # list of generators that need to yield new value
                postings = next_words[0][1]  # initialize postings of min nextWord, may merge if there is a match later
                i = 1
                while i < len(next_words):
                    if next_words[i][0] == next_words[i - 1][0]:
                        postings.extend(next_words[i][1])  # same word, merge postings
                        getNextVals.append(next_words[i][2])
                    else:
                        break
                    i += 1

                self.numberOfTokensProcessed += 1
                postings.sort(key=lambda x: x[0])  # sort postings by docid
                postings = self.tfidf_score(postings)  # change raw frequency to tf-idf
                keyValAsJson = {next_words[i-1][0]: postings}  # write k,v pair into final index

                json.dump(keyValAsJson, final)  # write the list of JSON objects to the file
                final.write("\n")  # add a newline character to the end of the file

                next_words = next_words[len(getNextVals):]  # remove written word

                for i, gen in enumerate(getNextVals):
                    try:
                        currGensNext = list(next(gen).items())
                        next_words.append((currGensNext[0][0], currGensNext[0][1], gen))
                    except StopIteration:
                        toBeRemoved.add(gen)
                file_generators = [gen for gen in file_generators if gen not in toBeRemoved]
                toBeRemoved = set()

        for file in self.partialIndexFiles:  # remove partial indices
            Path(file).unlink()
            # unindent onces???


    
    def print_stats(self):
        totalSize = os.stat("index.jsonl").st_size
        print("Number of files processed:", self.numberOfFilesProcessed)
        print("Number of unique tokens:", self.numberOfTokensProcessed)
        print("Total disk size (bytes):", totalSize)


if __name__ == "__main__":
    startTime = time.strftime("%H:%M:%S")
    print(startTime)
    index = Index(PATH)
    index.build_index()
    index.write_to_json()
    index.merge_files()
    index.write_important_words()
    index.print_stats()
    endTime = time.strftime("%H:%M:%S")
    print(endTime)
