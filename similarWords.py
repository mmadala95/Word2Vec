import os
import pickle
import re
import numpy as np
from scipy import spatial


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

words=['first','american','would']

def takeSecond(elem):
    return elem[1]

for word in words:
    v1 = embeddings[dictionary[word]]
    dotProduct = []
    for i in dictionary:
        v2=embeddings[dictionary[i]]
        # print (v2)
        dotProduct.append((i,spatial.distance.cosine(v1,v2)))
    # print(dotProduct)

    dotProduct=sorted(dotProduct,key=takeSecond)

    for index in range (20):
        print (dotProduct[index][0])

    print('------------')
