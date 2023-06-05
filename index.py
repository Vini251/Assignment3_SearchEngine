import os
import json
from pathlib import Path
import csv
from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk import stem
from urllib.parse import urldefrag
import sys
from math import log10
import time
from simhash import Simhash

STEMMER = stem.PorterStemmer()
PATH = "DEV/"
fullIndexPath = "index/index.csv"
importantTags = ["h1", "h2", "h3", "strong", "b"]
batch_size = 3000000
# Increase the field size limit
csv.field_size_limit(1024 * 1024 * 1024) 


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
        self.fingerPrints = []

    def write_important_words(self):
        with open("important_words.txt", 'w') as file:
            for word in self.important_words:
                file.write(word + '\n')

    def write_id_to_url(self):
        """Writing the idToUrl mapping to a CSV file."""
        with open("idToUrl.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "url"])
            for doc_id, url in self.idToUrl.items():
                writer.writerow([doc_id, url])


    def generate_line(self, filename):
        """This function reads the file and yields each JSON object."""
        with open(filename) as f:
            for line in f:
                obj = json.loads(line)
                yield obj


    def tfidf_score(self, postingsList: list):
        """Compute the TF-IDF score for postings"""
        freqToTfidf = []
        totalPostings = len(postingsList)

        for posting in postingsList:
            document_tfidf = posting.split(',')
            for dt in document_tfidf:
                document, frequency = dt.split(':')

                tf = 1 + log10(int(frequency))
                idf = log10(self.numberOfFilesProcessed / totalPostings)
                tfidf = round(tf * idf, 2)
                freqToTfidf.append((document, tfidf))

        return freqToTfidf



    
    def check_near_duplicaton(self, word_frequencies):
        sh = Simhash(word_frequencies)
        for fp in self.fingerPrints:
            score = sh.distance(fp)
            if score <= 0.9:
                return True
        self.fingerPrints.append(sh)
        return False

    def build_index(self):
        path = Path(self.filepath)
        for directory in path.iterdir():
            if not directory.is_dir():
                continue
            for file in directory.iterdir():
                if not file.is_file():
                    continue

                for obj in self.generate_line(file):
                    url = urldefrag(obj["url"]).url

                    if url in self.urlToId:
                        continue
                    doc_id = self.document_number
                    self.urlToId[url] = doc_id
                    self.idToUrl[doc_id] = url

                    raw_content = obj["content"]
                    soup = BeautifulSoup(raw_content, features="html.parser")

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

                    if self.check_near_duplicaton(frequency) == False: continue

                    for token, frequency in frequency.items():
                        if token not in self.inverted_index:
                            self.inverted_index[token] = []

                        if token in self.important_words:
                            self.inverted_index[token].append((doc_id, frequency * 100))
                        else:
                            self.inverted_index[token].append((doc_id, frequency))
                self.numberOfFilesProcessed += 1
                self.document_number += 1

                if sys.getsizeof(self.inverted_index) >= batch_size:
                    self.partial_index()

        if self.inverted_index:
            self.partial_index()

    def partial_index(self):
        path = f"index/partial_index_{self.partial_file_index}.csv"
        with open(path, "w", newline="") as partialFile:
            writer = csv.writer(partialFile)
            sorted_index = sorted(self.inverted_index.items(), key=lambda x: x[0])
            self.inverted_index = {}
            for key, value in sorted_index:
                postings = ', '.join(f"{doc_id}:{frequency}" for doc_id, frequency in value)
                writer.writerow([key, postings])
        self.partial_file_index += 1
        self.partialIndexFiles.append(path)


    def merge_files(self):
        """Opens the main index and all partial indices, sorts and writes all partial indices into separate files based on the first letter of the token."""
        # Open file readers for all partial index files
        file_readers = [csv.reader(open(filename)) for filename in self.partialIndexFiles]
        
        # Create data structures for file writers
        to_be_removed = set()
        file_writers = {}
        
        # Open the main index file for writing
        index_file_writer = csv.writer(open("index/index.csv", 'w', newline=''))
        index_file_writer.writerow(["token", "postings"])

        # Create separate file writers for each letter of the alphabet
        for i in range(26):
            letter = chr(ord('a') + i)
            filename = f"index/index_{letter}.csv"
            file_writers[letter] = csv.writer(open(filename, 'w', newline=''))
            file_writers[letter].writerow(["token", "postings"])

        next_rows = []

        # Read the first row from each file reader
        for i, reader in enumerate(file_readers):
            try:
                curr_row = next(reader)
                next_rows.append((curr_row[0], curr_row[1:], reader))
            except StopIteration:
                to_be_removed.add(reader)

        # Process the rows and write to the appropriate file
        while next_rows:
            next_rows.sort(key=lambda x: x[0].lower())  # Sort the rows case-insensitively
            
            # Get the first token and postings from the sorted rows
            get_next_vals = [next_rows[0][2]]
            postings = next_rows[0][1]
            i = 1
            while i < len(next_rows):
                if next_rows[i][0].lower() == next_rows[i - 1][0].lower():  # Compare tokens case-insensitively
                    postings.extend(next_rows[i][1])
                    get_next_vals.append(next_rows[i][2])
                else:
                    break
                i += 1

            self.numberOfTokensProcessed += 1
            token = next_rows[i - 1][0]

            postings.sort(key=lambda x: x[0])
            # Calculate TF-IDF scores for the postings
            postings_tfidf = self.tfidf_score(postings)

            # Create a list to store the formatted postings
            formatted_postings = []

            # Format the postings as "doc: tfidf" and append them to the list
            for posting in postings_tfidf:
                formatted_posting = f"{posting[0]}:{posting[1]}"
                formatted_postings.append(formatted_posting)

            # Combine the formatted postings into a single string
            postings_str = ", ".join(formatted_postings)

            # Write the token and postings to the appropriate file based on the first letter of the token
            if token[0].isalpha():
                first_letter = token[0].lower()
                file_writers[first_letter].writerow([token, postings_str])
            else:
                index_file_writer.writerow([token, postings_str])

            next_rows = next_rows[len(get_next_vals):]



            # Read the next row from the file readers
            for reader in get_next_vals:
                try:
                    curr_row = next(reader)
                    next_rows.append((curr_row[0], curr_row[1:], reader))
                except StopIteration:
                    to_be_removed.add(reader)

            # Remove the file readers that have reached the end of the file
            file_readers = [reader for reader in file_readers if reader not in to_be_removed]
            to_be_removed = set()

        # Remove the partial index files
        for file in self.partialIndexFiles:
            Path(file).unlink()




    def print_stats(self):
        total_size = 0
        for path, dirs, files in os.walk("index/"):
            for file in files:
                file_path = os.path.join(path, file)
                total_size += os.path.getsize(file_path)
        print("Number of files processed:", self.numberOfFilesProcessed)
        print("Number of unique tokens:", self.numberOfTokensProcessed)
        print("Total disk size (bytes):", total_size)


if __name__ == "__main__":
    startTime = time.strftime("%H:%M:%S")
    print(startTime)
    index = Index(PATH)
    index.build_index()
    index.write_id_to_url()
    index.merge_files()
    index.write_important_words()
    index.print_stats()
    endTime = time.strftime("%H:%M:%S")
    print(endTime)
