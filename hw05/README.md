# Task 5: Question answering
## Data
- MIRAGE collection https://github.com/nlpai-lab/MIRAGE
- SQuAD https://huggingface.co/datasets/rajpurkar/squad

## Tasks
In each case, calculate EM and F1 scores (SQuAD syle), EM-loose / EM-strict (MIRAGE style), and BERTScore.
1. Choose a small (<=4B parameters) LM (e.g., Qwen/Gemma/Phi). Evaluate model’s answers on MIRAGE in base (0-shot) mode. (15)
2. Extractive question answering (aka Machine Reading Comprehension) with a fine-tuned encoder model. (15) Find a BERT/RoBERTa model fine-tuned on SQuAD or NaturalQuestions from a reliable contributor on Huggingface. Apply the model to relevant passages (MIRAGE’s oracle mode).
3. Open-book question answering (MIRAGE’s oracle mode) with an instruction-tuned small LM. (15)
4. Use top-1 passage from two of your best rankers from HA4 as context for answer extraction/generation. (15)
5. Use concatenation of top-5 passages from two of your best rankers from HA4 as context for SLM. Compare with the results on mixed context from the MIRACLE dataset. (20)

Additional tasks (40)
6. Fine-tune the smallest variant of T5Gemma 2 model https://huggingface.co/google/t5gemma-2-270m-270m on
SQuAD data. Run experiments 1, 3, 4 (see above) using the model. (35)
7. Use LLM-as-a-judge approach to all obtained QA results. Analyze evaluation results. (5)