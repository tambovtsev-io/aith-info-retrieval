# Task 3: Learning to rank

## Data

1. Internet Mathematics 2009 dataset (attached)
2. WikIR collection (en1k sybset) https://github.com/getalp/wiIR
3. MIRAGE collection https://github.com/nlpai-lab/MIRAGE_

## Tasks

0.1 Refresh your knowledge of gradient boosting methods [http://www.chengli.io/tutorials/gradient_boosting.pdf](http://www.chengli.io/tutorials/gradient_boosting.pdf)

0.2 Read documentation and install CatBoost
https://catboost.ai/en/docs/

0.3 Read description of LETOR datasets from MSR
https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/

1 Apply CatBoost to MSR LETOR data following the tutorial, report results (10)
https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb

2. Learning to Rank on Internet Math 2009 data.

2.1 Re-format Internet mathematics 2009 data to comply with the MS LETOR format. Describe the data. (10)

2.2 Experiment with at least two methods, use NDCG as target metric. Describe the experiments and report results. (20)

3. Learning to rank on WikIR data.

3.1 An attempt to improve the BM25 ranking. (30)

- Think of features to add to BM25, for example: query length in words, distance between matched query terms in doc, etc., implement extraction of these features from query-document pairs.
- Take relevant document pairs + sample the same number of non-relevant documents for queries from the training set. Represent query-document pairs as vectors. Train a ranking function (motivate the choice of the method and target metric).
- Generate vectors for the top100 documents in BM25 rankings for test queries. Apply the learned ranking function to them. Evaluate the quality, compare with the results from assignment 2.

3.2 An attempt to “reconstruct” the BM25 ranking. (10)

- The BM25 score is a combination of a few features: 1) term frequencies of the query terms in the document, 2) inverse document frequencies of the query terms in the collection, 3) document length.
- Following the same procedure as in 3.1, represent query-document pairs as vectors, train a ranking function, evaluate it, and report results.

4. Experiments with the MIRAGE dataset (40)

4.1. Split the questions into train/test subsets (take into account the origin of the questions, see the paper).

4.2. Implement the following improvements:

- Each passage is represented as a two-field document (title + body).
- Collect pageview statistics for each passage’s page, see
    https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html
- Collect the number of incoming wiki links for each passage’s page, see
    https://linkcount.toolforge.org/api/
    alternatively get all in-links: https://www.mediawiki.org/wiki/API:Backlinks
- Combine these features with features from 3.1.

4.3. Train and evaluate a ranker, analyze and report results.