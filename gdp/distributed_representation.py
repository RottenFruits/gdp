import torch
import torch.optim as optim
import torch.utils.data
import collections
import numpy as np
from tqdm import tqdm
from gdp.word2vec  import word2vec

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
        
        x, y = self.model.generate_batch(self.corpus, self.window_size)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)

        for epo in range(num_epochs):
            loss_val = 0
            if self.ns == 1:
                ns = torch.LongTensor(np.array([self.model.negative_sampling(self.corpus) for i in range(len(x))]))
            if self.ns == 0:
                dataset = torch.utils.data.TensorDataset(x, y)
            else:
                dataset = torch.utils.data.TensorDataset(x, y, ns)

            batches = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

            for batch in batches:
                optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                loss_val += loss.data
                optimizer.step()
                        
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

        #vector = self.model.u_embeddings(word_idx).view(-1).detach().numpy()
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

