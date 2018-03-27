# Word-Embeddings

How do we represent word meaning so that we can analyze it, compare different words’ meanings, and use these representations in NLP tasks? One way to learn word meaning is to find regularities in how a word is used. Two words that appear in very similar contexts probably mean similar things. One way you could capture these contexts is to simply count which words appeared nearby. If we had a vocabulary of V words, we would end up with each word being represented as a vector of length jV j1 where for a word wi, each dimension j in wi’s vector, wi;j refers to how many times wj appeared in a context where wi was used. Word embeddings solve both of these problems by trying to encode the kinds of contexts a word appears in as a low-dimensional vector.


In word2vec.py I train a model to learn word representations using gradient decent and negative sampling and then try use those representations in intrinsic tasks that measure word similarity and an extrinsic task for sentiment analysis. 


The files:

1. unlabeled-text.tsv – Trained word2vec model on this data
2. extrinsic-train.tsv – This is the training data for the extrinsic evaluation (sentiment analysis).
3. extrinsic-dev.tsv – This is the development data for the extrinsic evaluation, to get a sense of performance
4. intrinsic-test.tsv – This is the data for the intrinsic evaluation on word similarity
5. extrinsic-test.tsv – This is the test data for the extrinsic evaluation
