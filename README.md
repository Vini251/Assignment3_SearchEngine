# CS121: Assignment_3 (Search Engine)

**Challenges:**
- Design efficient data structure
- Devise efficient file access
- Balance memory usage
- Response Time

**Team**:
- Vini Patel
- Alexander Butarita
- Oscar M
- Naveen Sastri
# **Milestone 1: Index (DUE: 5/12/2023)**

**Indexer:** 

- Create inverted index for corpus with data structures designed by you.  
- **Tokens:** all alphanumeric sequences in the dataset.  
- **Stop Words:** do not use stopping while indexing i.e. use all words, eventhe frequently occuring ones.  
- **Stemming:** use stemming for better textual matches. Suggestion: Porter Stemmer.  
- **Important text:** text in bold (with tags b and strong), in headings (tags h1, h2, h3), and in titles should be treated as more important than the ones in other places.

- Index should be stores in one or more files as partial indexes and then merged into one. (DO NOT USE DATABASE)
- Your indexer must off load the inverted index hash map from main memory to a partial index on disk at least 3 times during index construction; those partial indexes should be merged in the end.
- Optionally, after or during merging, they can also be split into separate index files with term ranges.
- The inverted index is simply a map with the token
as a key and a list of its corresponding postings. A posting is the representation
of the token’s occurrence in a document.
- The posting typically (not limited to)
contains the following info (you are encouraged to think of other attributes that
you could add to the index):
	- The document name/id the token was found in.
	- Its tf-idf score for that document (for MS1, add only the term frequency)
- Some tips:
	- When designing your inverted index, you will think about the structure
of your posting first.
	- You would normally begin by implementing the code to calculate/fetch
the elements which will constitute your posting.
	- Modularize. Use scripts/classes that will perform a function or a set of
closely related functions. This helps in keeping track of your progress,
debugging, and also dividing work amongst teammates if you’re in a group.
	- We recommend you use GitHub as a mechanism to work with your team
members on this project, but you are not required to do so.

# **Milestone 2: Search  (DUE: 5/19/2023)**      
- Your program should prompt the user for a query. **This doesn’t need to be a Web interface, it can be a console prompt.**     
- At the time of the query, your program will stem the query terms, look up your index, perform some calculations (see ranking below) and give out the ranked list of pages that are relevant for the query, with the most relevant on top.      
	- Ranking: at the very least, your ranking formula should include tf-idf,
and consider the important words, but you are free to add additional
components to this formula if you think they improve the retrieval
- Pages should be identified at least by their URLs (but you can use more informative representations).     
- The response to search queries should be ≤ 300ms. Ideally it would be . 100ms, but you won’t be penalized if it’s higher (as long as it’s kept ≤ 300ms).      
- Your search component must not load the entire inverted index in main memory. Instead, it must read the postings from the index(es) files on disk. The TAs will verify that this is happening. 
- Once you have built the inverted index, you are ready to test document retrieval
with queries. At the very least, the search should be able to deal with boolean
queries: AND only.
- If you wish, you can sort the retrieved documents based on tf-idf scoring
(you are not required to do so now, but doing it now may save you time in
the future). This can be done using the cosine similarity method. 
- Feel free to
use a library to compute cosine similarity once you have the term frequencies
and inverse document frequencies (although it should be very easy for you to
write your own implementation).
- You may also add other weighting/scoring
mechanisms to help refine the search results.

# **Milestone 3: Final Optimizationa and Demonstration  (DUE: 6/2/2023)** 
- Optimize the search engine.





# **How to run the search engine:**
- Download a zip file of this repository 
- To create inverted index, run this code on the termianl under the directory this repository is stored: python3 index.py
- To start the search engine, run this code on the termianl under the directory this repository is stored: python3 search.py

