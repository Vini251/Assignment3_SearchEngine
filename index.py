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

# Set up Porter Stemmer for word stemming
STEMMER = stem.PorterStemmer()

# Define the path to the documents
PATH = "DEV/"

# Define the path to the full index
fullIndexPath = "index/index.csv"

# Define important HTML tags
importantTags = ["h1", "h2", "h3", "strong", "b"]

# Set the batch size for partial index creation
batch_size = 3000000

# Increase the field size limit for CSV parsing
csv.field_size_limit(1024 * 1024 * 1024) 


class Index:
    def __init__(self, path) -> None:
        self.filepath = path
        self.idToUrl = {}  # Mapping of document ID to URL
        self.urlToId = {}  # Mapping of URL to document ID
        self.inverted_index = {}  # Inverted index for storing tokens and postings
        self.numberOfFilesProcessed = 0  # Number of files processed
        self.document_number = 0  # Number of documents
        self.numberOfTokensProcessed = 0  # Number of tokens processed
        self.partialIndexFiles = []  # List of paths to partial index files
        self.partial_file_index = 0  # Index for naming partial index files
        self.important_words = set()  # Set of important words
        self.fingerPrints = []  # List of Simhash fingerprints for near duplication detection

    def write_important_words(self):
        # Write the important words to a file
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
        frequency_tfidf = list()
        for post in postingsList:
            tfidf = round((1 + log10(post[1])) * (log10(self.numberOfFilesProcessed / len(postingsList))), 2)
            frequency_tfidf.append((post[0], tfidf))
        return frequency_tfidf
    
    def check_near_duplicaton(self, word_frequencies):
        """Check for near duplication using Simhash"""
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

                # Read each JSON object from the file
                for line in self.generate_line(file):
                    doc_id = self.document_number

                    # Store the URL in the idToUrl mapping
                    url = line["url"]
                    self.idToUrl[doc_id] = url
                    self.urlToId[url] = doc_id

                    # Pre-process and extract text from HTML
                    html = line["content"]
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Extract important words from specified HTML tags
                    for tags in soup.find_all(importantTags):
                        for words in re.sub(r"[^a-zA-Z0-9\s]", "", tags.text.lower()).split():
                            self.important_words.add(STEMMER.stem(words))

                    # Process and tokenize the text
                    soup = soup.find_all()
                    frequency = Counter()
                    for line in soup:
                        if not line.text:
                            continue
                        tokens = [STEMMER.stem(word) for word in re.sub(r"[^a-zA-Z0-9\s]", "", line.text.lower()).split()]
                        frequency.update(tokens)

                    # Check for near duplication
                    if self.check_near_duplicaton(frequency):
                        continue

                    # Update the inverted index with token frequencies
                    for token, frequency in frequency.items():
                        if token not in self.inverted_index:
                            self.inverted_index[token] = []

                        if token in self.important_words:
                            self.inverted_index[token].append((doc_id, frequency * 100))
                        else:
                            self.inverted_index[token].append((doc_id, frequency))
                            
                    self.numberOfFilesProcessed += 1
                    self.document_number += 1

                    # Create partial index files if the inverted index size exceeds the batch size
                    if sys.getsizeof(self.inverted_index) >= batch_size:
                        self.partial_index()

        # Create the final partial index if there are remaining tokens
        if self.inverted_index:
            self.partial_index()

        # Merge all partial index files into a single full index file
        self.merge_files()

        # Write the important words and idToUrl mapping to separate files
        self.write_important_words()
        self.write_id_to_url()

    def partial_index(self):
        """Write the inverted index to a partial index file"""
        path = f"index/partial_index_{self.partial_file_index}.csv"
        with open(path, "w", newline="") as partialFile:
            writer = csv.writer(partialFile)
            
            # Sort the inverted index by token
            sorted_index = sorted(self.inverted_index.items(), key=lambda x: x[0])
            
            # Clear the inverted index
            self.inverted_index = {}
            
            # Iterate over the sorted index
            for key, value in sorted_index:
                # Convert the postings list to a string representation
                postings = ', '.join(f"{doc_id}:{frequency}" for doc_id, frequency in value)
                
                # Write the token and postings to the file
                writer.writerow([key, postings])
        
        # Increment the partial file index and add the path to the list of partial index files
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
                else:
                    break
                i += 1

            self.numberOfTokensProcessed += 1
            token = next_rows[i - 1][0]
            
            # Write the token and postings to the appropriate file based on the first letter of the token
            if token[0].isalpha():
                first_letter = token[0].lower()
                postings_str = ", ".join(postings)  # Combine postings into a single string
                file_writers[first_letter].writerow([token, postings_str])
            else:
                postings_str = ", ".join(postings)  # Combine postings into a single string
                index_file_writer.writerow([token, postings_str])

            next_rows = next_rows[i:]

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
