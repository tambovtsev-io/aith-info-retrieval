# Task 2: BM25 ranking and search results evaluation

## Data

_WikIR collection_
https://github.com/getalp/wikIR
We’ll use the _en1k_ subset of the collection.

## Tasks

0. Preliminaries

0.1 Read carefully the paper describing the process of the WikIR construction:

_Jibril Frej, Didier Schwab, Jean-Pierre Chevallet. WIKIR: A Python Toolkit for Building a Large-scale Wikipedia-
based English Information Retrieval Dataset_
https://aclanthology.org/2020.lrec-1.237/

Understand how the queries and relevance judgments were obtained.

0.2 Install Rank-BM25 and read the documentation
https://github.com/dorianbrown/rank_bm

0.3 Install Scikit-learn (you’ll need tf.idf vectorizer)

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

0.4 Install ir-measures and read the documentation

https://ir-measur.es/en/latest/

0.5 Familiarize yourself with TREC runs and qrels formats.

1. Provide basic statistics for the _test_ queries (5)

```
- Number of queries
- query length in words
- Number of relevant docs per query
```

2. Run _test_ queries on three collection variants using tf.idf and Rank-BM25: original, stemmed, lemmatized
(don’t forget to process queries accordingly). Estimate the query execution time. Store the results in the TREC runs
format. (20)
3. Using ir-measures, evaluate your runs against qrels: calculate p@1, p@10, p@20, MAP, nDCG@
(provide the parameters of the measures, even if you use default one). Summarize the results in a table. (20)
4. Analyze this! (20)

For example:

```
- Are there “hard” and “easy” queries?
- How do query properties impact the performance?
- How does morphological processing influence the results?
- Compare evaluation measures. Would a single measure suffice?
- Inspect top-ranked documents judged non-relevant. Why are they in the top? Do you agree with the
judgments?
- Etc.
```
5. Using the _validation_ set, run BM25 with different _k1_ and _b_ parameters. Pick one measure to optimize, find the best
pair (k1, b). If it’s different from the default one, run BM25 with these parameters on the _test_ queries, report and
analyze the results. (15)


## Additional tasks

1. MIRAGE collection (20)

https://github.com/nlpai-lab/MIRAGE

1. Process the collection: collect all passages, generate qrels files.
2. Apply the same set of analyses and methods to the collection (note that there is no _validation_ subset).
3. Compare results on both collections.
2. Tolstoy & Dostoevsky (20)
1. Split both novels from the previous assignment into paragraphs.
2. Using BM25/tf.idf representations, find the most similar paragraphs within each novel and across two novels.
Which _idf_ scores would you use for this task? Analyze the results.


