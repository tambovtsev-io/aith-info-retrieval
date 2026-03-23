# Task 1: Analysis of text collections

## Data

_WikIR collection_
https://github.com/getalp/wikIR

We’ll use the _en1k_ subset of the collection. The collection is already cleaned, tokenized, and
lowercased. So far, we need only _documents.csv_ (the collection also contains automatically generated
queries; we’ll use them later).

_Stopwords_
https://gist.github.com/sebleier/

## Tasks

_Provide basic collection stats (20)_
1. Number of documents
2. Collection size in words
3. Avg. document length in words
4. Number of unique words (types)
5. Avg. word length
6. Avg. unique word (type) length

| Key | Value |
|-----|-------|
| Number of documents | 369721 (370k) |
| Collection size in words | 73093729 (73M) |
| Avg. document length in words | 197 |
| Number of unique words (types) | 794568 (800k) |
| Avg. word length | 4.8 |
| Avg. unique word (type) length | 7.7 |

_Build a frequency list (20)_

1. Build a word frequency list, i.e. a list of unique words in the collection along with their counts
    (frequencies), sorted by decreasing counts.
2. How many occurrences of stopwords are there in the collection? Do all top30 most frequent
    words occur in the stopword list? Would you recommend expanding the stopword list with
    some frequent words from the collection?
3. Draw a plot of ranks vs. collection frequencies in log-log coordinates (Zipf’s law).
4. Draw a plot of vocabulary growth (unique words) in log-log coordinates (Heaps’ law).

_Build a frequency list of word bigrams (15)_

1. How many unique bigrams are there in the collection?
2. Analyze the top of the list. Which bigrams would you keep as dictionary entries in the
    inverted index? Try to formalize the criteria.

_Morphological processing (25)_

1. Use NLTK’s implementation of the Porter stemmer
    https://www.nltk.org/api/nltk.stem.porter.html to stem the collection.
2. Install spaCy https://spacy.io/, download the smallest trained English model
    ( _en_core_web_sm_ ), see https://spacy.io/models/en. Lemmatize documents.
3. Apply BERT’s tokenizer to the collection, see
    https://huggingface.co/docs/transformers/v5.3.0/en/model_doc/bert
4. Provide basic stats for the stemmed/lemmatized/BERT-tokenized versions of the collection.

## Additional tasks

_Analysis of Tolstoy’s and Dostoevsky’s vocabularies (10)_

Compare the vocabulary growth rate in Leo Tolstoy’s _War and Peace_
(http://az.lib.ru/t/tolstoj_lew_nikolaewich/) and Fyodor Dostoevsky’s
_The Brothers Karamazov_ (http://az.lib.ru/d/dostoewskij_f_m/). Analyze both original and randomly
shuffled versions of the novels. Find each author’s most characteristic words.

_Detection of artificially generated texts (30)_

Apply the techniques from this assignments to the task of computer-generated text detection. Use
CoAT data for experiments (https://github.com/RussianNLP/CoAT).


