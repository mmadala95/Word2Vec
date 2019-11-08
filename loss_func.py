import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    matrixMul=tf.matmul(inputs,tf.transpose(true_w))
    A=tf.diag_part(matrixMul)
    B=tf.log(tf.reduce_sum(tf.exp(matrixMul),axis=1))
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here




    ==========================================================================
    """

    batchSize =labels.shape[0]
    biasSize=biases.shape[0]
    sampleSize=sample.shape[0]
    labels = tf.reshape(labels, [batchSize])
    biases = tf.reshape(biases, [biasSize, 1])
    samples = tf.convert_to_tensor(sample)
    unigramProb = tf.reshape(tf.convert_to_tensor(unigram_prob), [biasSize, 1])

    labelWeights=tf.nn.embedding_lookup(weights,labels)
    labelBiases=tf.nn.embedding_lookup(biases,labels)
    labelUnigram=tf.nn.embedding_lookup(unigramProb,labels)

    sampleWeights = tf.nn.embedding_lookup(weights, samples)
    sampleBiases = tf.nn.embedding_lookup(biases, samples)
    sampleUnigram = tf.nn.embedding_lookup(unigramProb, samples)


    s_wo=tf.add(tf.reshape(tf.diag_part(tf.matmul(inputs, tf.transpose(labelWeights))), [batchSize, 1]),labelBiases)
    s_wx=tf.add(tf.matmul(sampleWeights, tf.transpose(inputs)),sampleBiases)

    pr_wo=s_wo - tf.log((tf.scalar_mul(sampleSize, labelUnigram)) + 1e-10)
    pr_wx=s_wx - tf.log((tf.scalar_mul(sampleSize, sampleUnigram)) + 1e-10)


    termOne=tf.log_sigmoid(pr_wo)
    termTwo=tf.reduce_sum(tf.log((tf.ones(pr_wx.shape,)- tf.sigmoid(pr_wx))+ 1e-10), 0, keepdims=True)

    J=tf.scalar_mul(-1,((tf.transpose(termOne))+termTwo) )

    return J






