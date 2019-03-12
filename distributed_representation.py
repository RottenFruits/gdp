import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import collections
import numpy as np
from tqdm import tqdm

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
            self.model  = Skipgram(self.corpus.vocab_size, self.embedding_dim, self.negative_samples, self.sgns)
            if torch.cuda.is_available():
                self.model.cuda()
        elif mode_type == 2:
            print("2")
        
    def train(self, num_epochs = 100, learning_rate = 0.001):
        optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        for epo in range(num_epochs):
            loss_val = 0

            while self.model.batch_end  == 0:
                batches = self.model .generate_batch(self.corpus, self.window_size, self.batch_size)
                for i in tqdm(range(len(batches)), desc = 'batches', leave = False):
                    optimizer.zero_grad()
                    if self.sgns == 0:
                        loss = self.model(batches[i])
                    else:
                        loss = self.model(batches[i], self.corpus)
                    loss.backward()
                    loss_val += loss.data
                    optimizer.step()

            self.model.batch_end = 0
                        
            if self.trace == True:
                if epo % 10 == 0:
                    print(f'Loss at epo {epo}: {loss_val/len(batches)}')
                
    def get_vector(self, word):
        word_idx = self.corpus.dictionary[word]
        word_idx = torch.LongTensor([[ word_idx]])
        
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



class Skipgram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples, sgns):
        super(Skipgram, self).__init__()
        self.sgns = sgns
        self.negative_samples = negative_samples
        self.row_idx = 0
        self.col_idx = 0
        self.batch_end = 0
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        if sgns == 0:
            self.u_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim,  sparse = True)
            self.v_embeddings = torch.nn.Embedding(self.embedding_dim, self.vocab_size+1, sparse = True) 
        else:
            self.u_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim,  sparse = True)
            self.v_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim, sparse = True)
        #embedding init
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def generate_batch(self, corpus, window_size, batch_size):
        row_idx = self.row_idx
        col_idx = self.col_idx
        context = collections.deque()
        target  = collections.deque()
        i = 0

        while i < batch_size:
            data = corpus.data[row_idx]
            target_ = data[col_idx]
            sentence_length = len(data) 

            if col_idx == 0: #first word
                start_idx = col_idx + 1
                start_idx = 0 if  start_idx < 0 else start_idx
                end_idx = col_idx + 1 + window_size
                end_idx = end_idx if  end_idx < (sentence_length )  else sentence_length
                for t in range(start_idx, end_idx):
                    if t > sentence_length - 1:break
                    context.append(data[t])
                    target.append(target_)
                    i += 1
            elif col_idx == len(data): #last word
                start_idx = col_idx - window_size
                start_idx = 0 if  start_idx < 0 else start_idx
                end_idx = col_idx + 1
                end_idx = end_idx if  end_idx < (sentence_length)  else sentence_length 
                for t in range(start_idx, end_idx):
                    if t > sentence_length - 1:break
                    context.append(data[t])
                    target.append(target_)
                    i += 1
            else:#mid word
                start_idx = col_idx - window_size
                start_idx = 0 if  start_idx < 0 else start_idx
                end_idx = col_idx + 1 + window_size
                end_idx = end_idx if  end_idx < (sentence_length )  else sentence_length 
                for t in range(start_idx, end_idx):
                    if t > sentence_length - 1:break
                    if t == col_idx:continue
                    context.append(data[t])
                    target.append(target_)
                    i += 1

            col_idx = (col_idx + 1)
            if col_idx == len(data):
                col_idx  = 0
                row_idx = row_idx + 1
            
            if row_idx == len(corpus.data):
                self.row_idx = 0
                self.col_idx = 0
                self.batch_end = 1
                break
            else:
                self.row_idx = row_idx
                self.col_idx = col_idx
        
        return  np.vstack((np.array(target), np.array(context))).T

    def negative_sampling(self, corpus):
        sampled = np.random.choice(corpus.negaive_sample_table_w, p = corpus.negaive_sample_table_p, size = self.negative_samples)
        negative_samples = np.array([corpus.dictionary[w] for w in sampled])
        return negative_samples    

    def forward(self, batch, corpus = None):
        if self.sgns == 0:
            y_true = Variable(torch.from_numpy(np.array([batch[1]])).long())

            x1 = torch.LongTensor([[batch[0]]])
            x2 = torch.LongTensor([range(self.embedding_dim)])
            u_emb = self.u_embeddings(x1)
            v_emb = self.v_embeddings(x2)
            z = torch.matmul(u_emb, v_emb).view(-1) #view reshape

            log_softmax = F.log_softmax(z, dim = 0)
            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        else:        
            target = torch.LongTensor([[batch[0]]])
            context = torch.LongTensor([[batch[1]]])

            ns = self.negative_sampling(corpus)
            ns = torch.LongTensor([[ns]])

            if torch.cuda.is_available():
                target = Variable(target).cuda()
                context = Variable(context).cuda()
                ns = Variable(ns).cuda()

            #positive
            x1 = self.u_embeddings(target)
            x2 = self.v_embeddings(context)

            score = torch.sum(torch.mul(x1, x2))#inner product
            log_target = F.logsigmoid(score).squeeze()

            #negative
            x3 = self.v_embeddings(ns)
            neg_score = -1 * torch.sum(torch.mul(x1, x3), dim = 2)
            log_neg_sample = F.logsigmoid(neg_score).squeeze()

            loss = -1 * (log_target + log_neg_sample.sum())
    
        return loss