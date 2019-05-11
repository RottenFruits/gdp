import collections
import numpy as np

class Corpus:
    def __init__(self, data, mode, max_vocabulary_size = 5000, max_line = 0, minimum_freq = 0, tokenize = True, sub = 0):
        self.sub = sub
        
        if mode == "l":
            self.corpus = self.read_text_line(data)
            if max_line != 0: self.corpus = self.corpus[0:max_line]
            self.tokenized_corpus = self.tokenize_corpus(self.corpus)
            self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset_line(self.tokenized_corpus, max_vocabulary_size, minimum_freq)
        elif mode == "a":
            self.corpus = data
            if max_line != 0: self.corpus = self.corpus[0:max_line]
            if tokenize: 
                self.tokenized_corpus = self.tokenize_corpus(self.corpus)
            else:
                self.tokenized_corpus = self.corpus
            self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset_line(self.tokenized_corpus, max_vocabulary_size, minimum_freq)            
        elif mode == "o":
            self.tokenized_corpus = self.read_text(data)
            self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset(self.tokenized_corpus, max_vocabulary_size, minimum_freq)
            
        self.vocab_size = len(self.dictionary) - 1
        self.negaive_sample_table_w, self.negaive_sample_table_p = self.create_negative_sample_table()
        
 
    def read_text(self, file):
        data = []
        with open(file, 'r') as datafile:
            for line in datafile:
                data.append(line.lower().split(' '))
        return data[0]

    def read_text_line(self, file):
        import csv        
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                data.append(row[0])
        return data
                
    def build_dataset(self, words, vocabulary_size, minimum_freq):
        count = [['UNK', -1]]
        count = count[(count[:, 1].astype(int) >= minimum_freq) | (count[:, 1].astype(int) == -1)]
        count = count[count[:, 1].astype(int) >= minimum_freq]
        rho = int(np.percentile(count[1:, 1].astype(int), q = 0.90))
        if self.sub == 1:
            count = count[[self.subsampling(c, rho)[0] == 0 for _, c in count]]

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary
    
    def build_dataset_line(self, tokenized_corpus, vocabulary_size, minimum_freq):
        count = [['UNK', -1]]
        words = np.concatenate(tokenized_corpus)
        
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        count = np.array(count)
        count = count[(count[:, 1].astype(int) >= minimum_freq) | (count[:, 1].astype(int) == -1)]
        rho = int(np.percentile(count[1:, 1].astype(int), q = 0.90))
        if self.sub == 1:
            count = count[[self.subsampling(c, rho)[0] == 0 for _, c in count]]

        #dictionary
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        #data
        data = list()
        unk_count = 0
        for r in tokenized_corpus:
            data_tmp = list()
            for word in r:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
                data_tmp.append(index)
            data.append(data_tmp)
        count[0][1] = unk_count

        return data, count, dictionary, reverse_dictionary

    def tokenize_corpus(self, corpus):
        tokens = [np.array(x.split()) for x in corpus]
        return tokens
    
    def create_negative_sample_table(self):
        word_counts = self.count[1:, 1].astype(int) #index 0 is unk
        p = word_counts / word_counts.sum()
        p = np.power(p, 0.75)
        p /= p.sum()
        
        return np.array(list(range(1, len(self.dictionary)))), p
    
    def subsampling(self, count, rho = 1):
        if count == '-1':
            p = 0
        else:
            p = 1 - np.sqrt(rho / int(count))
        if p < 0:
            p = 0
        discard = np.random.binomial(1, p, 1)
        return discard