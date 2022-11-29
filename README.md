# SCAN: Scientific and Creative Analogies in Pretrained Language Models
---
# Intro

Code and dataset of our paper "Scientific and Creative Analogies in Pretrained Language Models", to appear in Findings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP). Find the paper preprint [here](https://arxiv.org/abs/2211.15268).


Human-like analogy making is more than traditional word analogies. When coming up with an analogy, humans usually:
1. Map an entire source domain to a target domain.
2. Map over large semantic distances.
3. Connect different types of concepts.

Traditional word analogy datasets suffer from:
1. Analogical relationships without much need for abstractive inference over the relational structure of the source and target domains.
2. In-domain words are usually different instances from the same concept (e.g. all capitals).
3. The chances of the words in the source and target domains co-occurring in the training corpus of a pretrained language model are quite high, meaning less analogical abstraction needs to be made to recognize their relationship.

We propose SCAN, a new evaluation dataset which addresses these points and offers a more holistic collection of analogies. We systematically evaluate BERT, GPT-2, Multilingual BERT and a GloVe baseline on these challenging new analogies and find their performance to be severely lacking.

If you use this code or dataset, please cite us!
