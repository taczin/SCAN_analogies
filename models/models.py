import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertModel, GPT2Model
import torch
import numpy as np
from _collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GloveNoTraining(nn.Module):
    def __init__(self, config):
        import faiss
        super(GloveNoTraining, self).__init__()
        self.idx2word = ['UNK', 'PAD']
        self.word2idx = defaultdict(int)
        self.word2idx['UNK'] = 0
        self.word2idx['PAD'] = 1
        idx = 2
        vectors = [np.zeros(300), np.ones(300)]
        cnt = 0
        with open(f'models/glove.6B.300d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.idx2word.append(word)
                self.word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:], dtype=np.float32)
                vectors.append(vect)
                cnt += 1
        self.embedding_np = np.stack(vectors, axis=0).astype(np.float32)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embedding_np), freeze=True)
        print('done loading predictive glove without training')
        self.index = faiss.IndexFlatL2(self.embedding.embedding_dim)  # build the index
        self.index.add(self.embedding_np)

    def forward(self, batch):
        subtractions = torch.zeros((len(batch), self.embedding.embedding_dim))
        # need this awkward loop setup since the expressions in the triples are not single words but also sometimes
        # consist of multiple words whose vectors then need to be averaged
        embedded = torch.zeros((3, len(batch), self.embedding.embedding_dim))
        for batch_id in range(len(batch)):
            for expr_id, expr in enumerate(batch[batch_id]):
                expr_vec = torch.zeros((self.embedding.embedding_dim)).to(self.embedding.weight.device.type)
                cnt = 0
                for word in expr:
                    expr_vec += self.embedding(torch.tensor(word).to(self.embedding.weight.device.type))
                    cnt += 1
                expr_vec = expr_vec/cnt
                embedded[expr_id, batch_id] = expr_vec

        # embedded = self.embedding(batch)
        subtraction = torch.abs(embedded[1] - embedded[0]) + embedded[2]

        return subtraction

    def get_state_dict(self):
        dicts = {'embedding': self.embedding.state_dict()}
        return dicts

    def load_model(self, params):
        self.embedding.load_state_dict(params['embedding'])
