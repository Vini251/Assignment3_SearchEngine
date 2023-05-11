import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Example
# inverted_index = {
#   "apple": [("doc1", 3), ("doc2", 1), ("doc3", 2)],
#   "orange": [("doc1", 1), ("doc2", 2), ("doc4", 1)],
# }

def tokenize(url):
    # Retrieve HTML content from the URL
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text from the HTML content
    text = soup.get_text()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Count the frequency of each token
    token_freq = nltk.FreqDist(tokens)

    return token_freq