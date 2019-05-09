import torch
from torch.autograd import Variable
import torch.nn.functional as F
import collections
import numpy as np

class word2vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples, ns, model_type):
        super(word2vec, self).__init__()
        self.ns = ns
        self.negative_samples = negative_samples
        self.row_idx = 0
        self.col_idx = 0
        self.batch_end = 0
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size + 1
        self.model_type = model_type
        
        self.u_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim, sparse = True)
        self.v_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim, sparse = True)

        #if ns == 0:
        #    self.u_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim,  sparse = True)
        #    self.v_embeddings = torch.nn.Embedding(self.embedding_dim, self.vocab_size+1, sparse = True) 
        #else:
        #    self.u_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim, sparse = True)
        #    self.v_embeddings = torch.nn.Embedding(self.vocab_size+1, self.embedding_dim, sparse = True)

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

            start_idx = col_idx - window_size
            start_idx = 0 if  start_idx < 0 else start_idx
            end_idx = col_idx + 1 + window_size
            end_idx = end_idx if  end_idx < (sentence_length )  else sentence_length

            if self.model_type == "skip-gram":
                for t in range(start_idx, end_idx):
                    if t > sentence_length - 1:break
                    if t == col_idx:continue
                    context.append(data[t])
                    target.append(target_)
                    i += 1
            else:
                context.append([data[x] for i, x in enumerate(range(start_idx, end_idx)) if x != col_idx])
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
        
        if self.model_type == "skip-gram":
            batches = np.vstack((np.array(target), np.array(context))).T
        elif self.model_type == "cbow":
            batches = []
            for c, t in zip(context, target):
                batches.append([c, t])
            batches = np.array(batches)
        return batches

    def generate_batch2(self, corpus, window_size):
        row_idx = self.row_idx
        col_idx = self.col_idx
        context = collections.deque()
        target  = collections.deque()
        i = 0

        while row_idx < len(corpus.data):

            data = corpus.data[row_idx]
            target_ = data[col_idx]
            sentence_length = len(data)

            start_idx = col_idx - window_size
            start_idx = 0 if  start_idx < 0 else start_idx
            end_idx = col_idx + 1 + window_size
            end_idx = end_idx if  end_idx < (sentence_length )  else sentence_length

            if self.model_type == "skip-gram":
                for t in range(start_idx, end_idx):
                    if t > sentence_length - 1:break
                    if t == col_idx:continue
                    context.append(data[t])
                    target.append(target_)
                    i += 1
            else:
                context.append([data[x] for i, x in enumerate(range(start_idx, end_idx)) if x != col_idx])
                target.append(target_)
                i += 1
                
            col_idx = (col_idx + 1)
            if col_idx == len(data):
                col_idx  = 0
                row_idx = row_idx + 1
                    
        if self.model_type == "skip-gram":
            batches = np.vstack((np.array(target), np.array(context))).T
        elif self.model_type == "cbow":
            batches = []
            for c, t in zip(context, target):
                batches.append([c, t])
            batches = np.array(batches)
        return batches

    def negative_sampling(self, corpus):
        sampled = np.random.choice(corpus.negaive_sample_table_w, p = corpus.negaive_sample_table_p, size = self.negative_samples)
        negative_samples = np.array([corpus.dictionary[w] for w in sampled])
        return negative_samples    

    def forward(self, batch, corpus = None):
        if self.model_type == "skip-gram":
            if self.ns == 0:
                #y_true = Variable(batch[1])
                y_true = batch[1]
                x1 = batch[0]
                x2 = torch.LongTensor(range(self.vocab_size))
                    
                u_emb = self.u_embeddings(x1)
                v_emb = self.v_embeddings(x2)
                z = torch.matmul(u_emb, torch.t(v_emb))
                
                log_softmax = F.log_softmax(z, dim = 1)
                loss = F.nll_loss(log_softmax, y_true)

            else:        
                x1 = torch.LongTensor([[batch[0]]])
                x2 = torch.LongTensor([[batch[1]]])

                ns = self.negative_sampling(corpus)
                ns = torch.LongTensor([[ns]])

                #positive
                u_emb = self.u_embeddings(x1)
                v_emb = self.v_embeddings(x2)

                score = torch.sum(torch.mul(u_emb, v_emb))#inner product
                log_target = F.logsigmoid(score).squeeze()

                #negative
                v_emb_negative = self.v_embeddings(ns)
                neg_score = -1 * torch.sum(torch.mul(u_emb, v_emb_negative), dim = 2)
                log_neg_sample = F.logsigmoid(neg_score).squeeze()

                loss = -1 * (log_target + log_neg_sample.sum())
        elif self.model_type == "cbow":
            if self.ns == 0:
                y_true = Variable(torch.from_numpy(np.array([batch[1]])).long())
                x1 = torch.LongTensor(batch[0])
                x2 = torch.LongTensor(range(self.embedding_dim))
                    
                u_emb = torch.mean(self.u_embeddings(x1), dim = 0)
                v_emb = self.v_embeddings(x2)
                z = torch.matmul(u_emb, v_emb).view(-1)

                log_softmax = F.log_softmax(z, dim = 0)
                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            else:
                x1 = torch.LongTensor(batch[0])
                x2 = torch.LongTensor([batch[1]])

                ns = self.negative_sampling(corpus)
                ns = torch.LongTensor([ns])

                #positive
                u_emb = torch.mean(self.u_embeddings(x1), dim = 0)
                v_emb = self.v_embeddings(x2)

                score = torch.sum(torch.mul(u_emb, v_emb))#inner product
                log_target = F.logsigmoid(score).squeeze()

                #negative
                v_emb_negative = self.v_embeddings(ns)
                neg_score = -1 * torch.sum(torch.mul(u_emb, v_emb_negative), dim = 2)
                log_neg_sample = F.logsigmoid(neg_score).squeeze()

                loss = -1 * (log_target + log_neg_sample.sum())    
        return loss
