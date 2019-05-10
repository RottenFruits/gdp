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
        if torch.cuda.is_available():
            self.u_embeddings = self.u_embeddings.cuda()
            self.v_embeddings = self.v_embeddings.cuda()

        #embedding init
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def generate_batch(self, corpus, window_size):
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
                c = [data[x] for i, x in enumerate(range(start_idx, end_idx)) if x != col_idx]
                if len(c) == (window_size * 2):
                    context.append(c)
                    target.append(target_)
                i += 1

            
            col_idx = (col_idx + 1)
            if col_idx == len(data):
                col_idx  = 0
                row_idx = row_idx + 1
                    
        if self.model_type == "skip-gram":
            x = np.array(target)
            y = np.array(context)
        elif self.model_type == "cbow":
            x = np.array(context)
            y = np.array(target)
        return x, y

    def negative_sampling(self, corpus):
        negative_samples = np.random.randint(low = 1, high = self.vocab_size, size = self.negative_samples)
        #negative_samples = np.random.choice(corpus.negaive_sample_table_w, p = corpus.negaive_sample_table_p, size = self.negative_samples)
        return negative_samples    

    def forward(self, batch, corpus = None):
        if self.model_type == "skip-gram":
            if self.ns == 0:                    
                u_emb = self.u_embeddings(batch[0])
                v_emb = self.v_embeddings(torch.LongTensor(range(self.vocab_size)))
                z = torch.matmul(u_emb, torch.t(v_emb))
                log_softmax = F.log_softmax(z, dim = 1)
                loss = F.nll_loss(log_softmax, batch[1])
            else:
                #positive
                u_emb = self.u_embeddings(batch[0])
                v_emb = self.v_embeddings(batch[1])
                score = torch.sum(torch.mul(u_emb, v_emb), dim = 1)#inner product
                log_target = F.logsigmoid(score)

                #negative
                v_emb_negative = self.v_embeddings(batch[2])
                neg_score = -1 * torch.sum(torch.mul(u_emb.view(batch[0].shape[0], 1, self.embedding_dim), v_emb_negative.view(batch[0].shape[0], batch[2].shape[1], self.embedding_dim)), dim = 2)
                log_neg_sample = F.logsigmoid(neg_score)

                loss = -1 * (log_target.sum() + log_neg_sample.sum())
        elif self.model_type == "cbow":
            if self.ns == 0:
                u_emb = torch.mean(self.u_embeddings(batch[0]), dim = 1)
                v_emb = self.v_embeddings(torch.LongTensor(range(self.vocab_size)))
                z = torch.matmul(u_emb, torch.t(v_emb))

                log_softmax = F.log_softmax(z, dim = 1)
                loss = F.nll_loss(log_softmax, batch[1])
            else:
                #positive
                u_emb = torch.mean(self.u_embeddings(batch[0]), dim = 1)
                v_emb = self.v_embeddings(batch[1])

                score = torch.sum(torch.mul(u_emb, v_emb), dim = 1)#inner product
                log_target = F.logsigmoid(score)

                #negative
                v_emb_negative = self.v_embeddings(batch[2])
                neg_score = -1 * torch.sum(torch.mul(u_emb.view(batch[0].shape[0], 1, self.embedding_dim), v_emb_negative.view(batch[0].shape[0], batch[2].shape[1], self.embedding_dim)), dim = 2)
                log_neg_sample = F.logsigmoid(neg_score)

                loss = -1 * (log_target.sum() + log_neg_sample.sum())
        return loss
