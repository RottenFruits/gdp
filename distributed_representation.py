import torch
import torch.optim as optim
import collections
import numpy as np
from tqdm import tqdm
import skipgram as sk

class DistributedRepresentation:
    def __init__(self, corpus, embedding_dim, window_size, batch_size, mode_type = 1, 
                 negative_samples = 10, sgns = 0, trace = False):
        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.sgns = sgns
        self.negative_samples = negative_samples
        self.trace = trace

        #model set
        if mode_type == 1:
            self.model  = sk.Skipgram(self.corpus.vocab_size, self.embedding_dim, self.negative_samples, self.sgns)
            if torch.cuda.is_available():
                self.model.cuda()
        elif mode_type == 2:
            print("2")
        
    def train(self, num_epochs = 100, learning_rate = 0.001):
        optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        for epo in range(num_epochs):
            loss_val = 0

            while self.model.batch_end  == 0:
                batches = self.model.generate_batch(self.corpus, self.window_size, self.batch_size)
                optimizer.zero_grad()
                if self.sgns == 0:
                    loss = self.model(batches)
                else:
                    loss = self.model(batches, self.corpus)
                loss.backward()
                loss_val += loss.data.item()
                optimizer.step()

            self.model.batch_end = 0
                        
            if self.trace == True:
                if epo % 10 == 0:
                    print(f'Loss at epo {epo}: {loss_val/len(batches)}')
                
    def get_vector(self, word):
        word_idx = self.corpus.dictionary[word]
        word_idx = torch.LongTensor([[word_idx]])
        if torch.cuda.is_available():
            word_idx = word_idx.cuda()
            vector = self.model.u_embeddings(word_idx).view(-1).detach().cpu().numpy()
        else:
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

