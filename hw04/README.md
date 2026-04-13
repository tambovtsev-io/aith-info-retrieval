# Task 4: Neural retrieval and re-ranking

## Data

_WikIR collection (en1k subset) https://github.com/getalp/wikIR
MIRAGE collection_ https://github.com/nlpai-lab/MIRAGE

## Tasks

0. Install SentenceTransformers

https://www.sbert.net/
https://huggingface.co/sentence-transformers

Read the documentation, look at the examples, pay special attention to the search section:

https://www.sbert.net/examples/applications/semantic-search/README.html
https://www.sbert.net/examples/applications/retrieve_rerank/README.html

1. Ranking (25)

Choose at least two pre-trained models for your experiments
https://www.sbert.net/docs/sentence_transformer/pretrained_models.html. Motive the choice of the models.

Perform retrieval experiments using the WikIR and MIRAGE test sets.

(Have a look at these utils:
https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.semantic_search )

2. Re-ranking (25)

Re-rank top-k BM25 results using pre-trained neural models. Motivate the choice of the model. Perform experiments using using the WikIR and MIRAGE test sets (experiment with different k). Evaluate the efficiency of the configuration.

3. Mixture model (30)

Optimize a mixture model $alpha*BM25 + (1-alpha)*q\_d\_cosine\_similarity$ based on WikIR train set: find an _alpha_ that maximizes a chosen IR measure_._ (Min-max normalize BM25 scores, so they are in the range [0,1].) Apply the formula to the WikIR and MIRAGE test sets.

For all configurations (1—3), calculate p@1, p@10, p@20, MAP@20, nDCG@20. Summarize the results in a table (add results from home assignments 2 & 3), analyze the results.

_Additional task (40)_

Fine-tune a cross-encoder on the WikiIR training data. Carefully describe your training configuration. Apply the fine-tuned model to both WikIR and MIRAGE test sets. Analyze results.