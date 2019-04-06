import torch
import torch.optim as optim
import collections
import numpy as np
from tqdm import tqdm
from word2vec  import word2vec

class DistributedRepresentation:
    def __init__(self, corpus, embedding_dim, window_size, batch_size, model_type = "skip-gram", 
                 negative_samples = 10, ns = 0, trace = False):
        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.ns = ns
        self.negative_samples = negative_samples
        self.trace = trace
        self.model_type = model_type

        self.model  = word2vec(self.corpus.vocab_size, self.embedding_dim, self.negative_samples, self.ns, self.model_type)
        
    def train(self, num_epochs = 100, learning_rate = 0.001):
        optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        for epo in range(num_epochs):
            loss_val = 0

            while self.model.batch_end  == 0:
                batches = self.model.generate_batch(self.corpus, self.window_size, self.batch_size)
                for i in tqdm(range(len(batches)), desc = 'batches', leave = False):
                    optimizer.zero_grad()
                    if self.ns == 0:
                        loss = self.model(batches[i])
                    else:
                        loss = self.model(batches[i], self.corpus)
                    loss.backward()
                    loss_val += loss.data
                    optimizer.step()

            self.model.batch_end = 0
            #shuffle
            rind = np.random.permutation(len(self.corpus.data))
            self.corpus.data = np.array(self.corpus.data)[rind]
                        
            if self.trace == True:
                if epo % 10 == 0:
                    print(f'Loss at epo {epo}: {loss_val/len(batches)}')
                
    def get_vector(self, word):
        word_idx = self.corpus.dictionary[word]
        word_idx = torch.LongTensor([[word_idx]])
        vector = self.model.u_embeddings(word_idx).view(-1).detach().numpy()
        return vector

    def similarity_pair(self, word1, word2):
        return np.dot(self.get_vector(word1), self.get_vector(word2)) / (np.linalg.norm(self.get_vector(word1)) * np.linalg.norm(self.get_vector(word2)))
    
    def similarity(self, word, descending = True):
        words = np.array([x for x in self.corpus.dictionary.items()])
        
        sim = np.array(list(map(lambda x: self.similarity_pair(word, x[0]), words)))#calculate similarity
        sim_list = np.vstack((sim, words[:, 0])).T
        
        if descending:
            rnk = np.argsort(sim, )[::-1]
        else:
            rnk = np.argsort(sim, )
    
        sim_list[rnk]    
        return sim_list[rnk]

