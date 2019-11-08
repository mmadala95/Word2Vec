import os
import pickle
import re as regex
import numpy as np


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
outputFile = open("word_analogy_predictions_nce.txt", "w")
readFile = open("word_analogy_dev.txt", "r")
sentences = readFile.readlines()
regexPattern = '|'.join(map(regex.escape, [",", "||"]))
for sentence in sentences:
    parts = regex.split(regexPattern, sentence.strip() )
    difference = []
    for part in parts[:3]:
        part = part.replace("\"", "")
        words = regex.split('|'.join(map(regex.escape, [":"])), part)
        v1, v2 = words
        v1 = embeddings[dictionary[v1]]
        v2 = embeddings[dictionary[v2]]
        difference.append(np.subtract(v2, v1))
    vectorDifference = np.mean(difference, axis=0)

    closeness = {}
    for part in parts[3:]:
        outputFile.write(part + " ")
        words = regex.split('|'.join(map(regex.escape, [":"])), part.replace("\"", ""))
        v1, v2 = words
        v1 = embeddings[dictionary[v1]]
        v2 = embeddings[dictionary[v2]]
        next_index=len(closeness)
        closeness[next_index] = np.dot(np.subtract(v2, v1), vectorDifference) / (np.linalg.norm(np.subtract(v2, v1)) * np.linalg.norm(vectorDifference))
    minIndex = min(closeness, key=closeness.get) + 3
    maxIndex = max(closeness, key=closeness.get) + 3
    outputFile.write(parts[minIndex] + " " + parts[maxIndex] + "\n")
readFile.close()
outputFile.close()

