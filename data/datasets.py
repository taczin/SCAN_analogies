from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer, GPT2Tokenizer
import random
import faiss
from collections import defaultdict
import pickle as pkl

def set_seeds(seed_int):
    global seed
    seed = seed_int
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class Glove:
    def __init__(self):
        self.idx2word = ['UNK', 'PAD']
        self.word2idx = defaultdict(int)
        self.word2idx['UNK'] = 0
        self.word2idx['PAD'] = 1
        idx = 2
        vectors = [np.zeros(300), np.ones(300)]
        with open(f'models/glove.6B.300d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.idx2word.append(word)
                self.word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:], dtype=np.float32)
                vectors.append(vect)
        self.embedding = np.stack(vectors, axis=0).astype(np.float32)


    def init_index(self):
        d = len(self.embedding[0])  # dimension
        self.index = faiss.IndexFlatL2(d)  # build the index
        # print(index.is_trained)
        self.index.add(self.embedding)  # add vectors to the index

    def find_k_nearest_neighbors(self, query_words, k):
        d = len(self.embedding[0])
        xq = np.zeros((len(query_words), d), dtype='float32')
        for i, word in enumerate(query_words):
            #if word == 'applications':
            #    print('applications')
            if isinstance(word, list):
                tmp = word[0]
            else:
                tmp = word
            xq[i] = self.embedding[self.word2idx[tmp]]


        #print(index.ntotal)

        l = k+4  # we want to see 1 nearest neighbors, k=2 since most #1 neighbor is word itself
        D, I = self.index.search(xq, l)  # actual search
        #print(I[:5])  # neighbors of the 5 first queries
        #print(I[-5:])  # neighbors of the 5 last queries
        distractors = []
        for i, row in enumerate(I):
            #per_word_distractors = []
            if isinstance(query_words[i], list):
                query_idx = [self.word2idx[word] for word in query_words[i]]
            else:
                query_idx = np.array([self.word2idx[query_words[i]]])
            words = np.setdiff1d(row, query_idx)[:k]
            words = [self.idx2word[idx] for idx in words]
            #for idx in row:
            #    if isinstance(query_words[i], list):
            #        if self.idx2word[idx] not in query_words[i]:
            #            per_word_distractors.append(self.idx2word[idx])
            #    else:
            #        if self.idx2word[idx] != query_words[i]:
            #            per_word_distractors.append(self.idx2word[idx])
            distractors.append(words)
        return distractors



def make_combinations(pairs, alts=False, distractors=False):
    max_alt_len = 267
    quadruples = []
    combinations = np.array(
        np.meshgrid(np.arange(len(pairs)), np.arange(len(pairs)))).T.reshape(-1, 2)
    for i in range(len(combinations)):
        # if a pair isn't combined with itself
        if combinations[i, 0] != combinations[i, 1]:
            # if there are also alternative answers:
            if isinstance(pairs.iloc[combinations[i, 0], 1], list):
                tmp = pairs.iloc[combinations[i, 0], 1][0]
            else:
                tmp = pairs.iloc[combinations[i, 0], 1]
            if isinstance(pairs.iloc[combinations[i, 1], 1], list):
                word4 = pairs.iloc[combinations[i, 1], 1][0]
                alternatives = pairs.iloc[combinations[i, 1], 1][1:]
                candidates = pairs.iloc[combinations[i, 1], 2].copy()
                if len(candidates) < 4:
                    fill = 4 - len(candidates)
                    for i in range(fill):
                        candidates.append('none')
                # print(len(candidates))
                rand_id = np.random.randint(0, len(candidates) + 1)
                label = rand_id
                candidates.insert(rand_id, word4)
                quadruples.append([pairs.iloc[combinations[i, 0], 0], tmp,
                                   pairs.iloc[combinations[i, 1], 0], word4, word4, label,
                                   candidates, alternatives, pairs.iloc[0]['rel'], pairs.iloc[0]['subrel']])

                if alts:
                    for altn in alternatives:
                        word4 = altn
                        quadruples.append([pairs.iloc[combinations[i, 0], 0], tmp,
                                           pairs.iloc[combinations[i, 1], 0], word4, word4, 1,
                                           pairs.iloc[0]['rel'], pairs.iloc[0]['subrel']])
                else:
                    filler = max_alt_len - len(alternatives)
                    alternatives.extend(filler * ['None'])
            else:
                alternatives = max_alt_len * ['None']
                word4 = pairs.iloc[combinations[i, 1], 1]
                candidates = pairs.iloc[combinations[i, 1], 2].copy()
                if len(candidates) < 4:
                    fill = 4 - len(candidates)
                    for i in range(fill):
                        candidates.append('none')
                # print(len(candidates))
                rand_id = np.random.randint(0, len(candidates) + 1)
                candidates.insert(rand_id, word4)
                label = rand_id
                quadruples.append([pairs.iloc[combinations[i, 0], 0], tmp,
                                   pairs.iloc[combinations[i, 1], 0], word4, word4, label,
                                   candidates, alternatives, pairs.iloc[0]['rel'], pairs.iloc[0]['subrel']])
            # add distractor examples
            #if word4 in pairs.iloc[combinations[i, 1]]['distractors']:
            #    print('found it')
            # comment this in for old distractor version
            #if distractors:
            #    alternatives = max_alt_len * ['None']
            #    for word in pairs.iloc[combinations[i, 1]]['distractors']:
            #        quadruples.append([pairs.iloc[combinations[i, 0], 0], tmp,
            #                           pairs.iloc[combinations[i, 1], 0], word, word4, 0,
            #                           alternatives, pairs.iloc[0]['rel'], pairs.iloc[0]['subrel']])

    df_quadruples = pd.DataFrame.from_records(quadruples, columns=['word1', 'word2', 'word3', 'word4', 'original',
                                                               'label', 'distractors', 'alternatives', 'relation', 'subrelation'])
    return df_quadruples

def create_strict_bats_split(mode='strict', alts=False, distractors=False, meta_split=False, seed_int=101, test_shots=4):
    """
    Creates an instance of the BATS dataset. If init is called with these parameters, then the train, dev and test
    splits are created and saved as files. To load them, use the other init function.
    """
    set_seeds(seed_int)
    print('Loading BATS dataset...')
    folder_path = 'bats'
    quadruples = []
    glove = Glove()
    glove.init_index()
    train = []
    dev = []
    test = []
    support = []
    query = []
    with os.scandir('data/' + folder_path + '/') as relations:
        for rel in relations:
            if os.path.isdir('data/{}/{}/'.format(folder_path, rel.name)):
                with os.scandir('data/{}/{}/'.format(folder_path, rel.name)) as subrelations:
                    for subrel in subrelations:
                        pairs = pd.read_table('data/{}/{}/{}'.format(folder_path, rel.name, subrel.name), header=None)
                        pairs[1] = pairs[1].apply(lambda x: x.split('/') if '/' in x else x)
                        if distractors:
                            pairs['distractors'] = glove.find_k_nearest_neighbors(pairs[1].tolist(), 4)
                        pairs['rel'] = rel.name
                        pairs['subrel'] = subrel.name
                        if mode == 'all':
                            combinations = make_combinations(pairs, alts, distractors)
                            train.append(combinations)
                            continue
                        else:
                            # if we need a split into train, dev and test for meta learning, dev and test splits need
                            # to contain unseen relations
                            if meta_split:
                                df_support = pairs.sample(frac=0.5, random_state=seed)
                                df_query = pairs[~pairs.index.isin(df_support.index)]
                                support_combinations = make_combinations(df_support, alts, distractors=distractors)
                                query_combinations = make_combinations(df_query, alts, distractors=distractors)
                                support.append(support_combinations)
                                query.append(query_combinations)
                            else:
                                # split each file into a train, a dev and a test part
                                df_train = pairs.sample(frac=0.7, random_state=seed)
                                tmp = pairs[~pairs.index.isin(df_train.index)]
                                # set seed +1 to prevent resuing the same idx as one line before
                                df_dev = tmp.sample(frac=0.5, random_state=seed+1)
                                df_test = tmp[~tmp.index.isin(df_dev.index)]
                                # make the analogy quadruples from each split
                                train_combinations = make_combinations(df_train, alts, distractors=distractors)
                                dev_combinations = make_combinations(df_dev, alts, distractors=distractors)
                                test_combinations = make_combinations(df_test, alts, distractors=distractors)
                                train.append(train_combinations)
                                dev.append(dev_combinations)
                                test.append(test_combinations)



    if mode == 'all':
        df = pd.concat(train)
        sufx = '_alts' if alts else ''
        df = df.dropna()
        df.to_csv('data/bats_all{}.csv'.format(sufx))
        print(len(df))
    elif meta_split:
        rels = {}
        df_sup = pd.concat(support)
        df_qry = pd.concat(query)
        relation_types = list(df_sup['subrelation'].unique())
        rels['train'] = random.sample(relation_types, 30)
        tmp = set(relation_types) - set(rels['train'])
        rels['dev'] = random.sample(tmp, 5)
        rels['test'] = set(tmp) - set(rels['dev'])
        for split in ['train', 'dev', 'test']:
            df1 = df_sup.loc[df_sup['subrelation'].isin(rels[split])]
            df1 = df1.sample(frac=1, random_state=seed+2).reset_index(drop=True)
            df2 = df_qry.loc[df_qry['subrelation'].isin(rels[split])]
            df2 = df2.sample(frac=1, random_state=seed+3).reset_index(drop=True)
            #df = pd.concat([df1, df2])
            sufx1 = '_strict'
            sufx2 = '_alts' if alts else ''
            sufx3 = '_distractors' if distractors else ''
            sufx4 = '_meta' if meta_split else ''
            # testing is different to validation in the sense that for validation, for each task, I sample k shots
            # from the support set and evaluate on k samples from the query set. This is done as many times as there
            # are enough samples in both sets and the results averaged for each task. For testing, I only sample k shots
            # from the support data once for each task and evaluate on all the rest.
            if split == 'test':
                sup_dfs = []
                qry_dfs = [df2]
                for rel in rels['test']:
                    tmp = df1.loc[df1['subrelation']==rel]
                    sup_df= tmp.sample(test_shots, random_state=seed_int)
                    qry_df = tmp[~tmp.index.isin(sup_df.index)]
                    sup_dfs.append(sup_df)
                    qry_dfs.append(qry_df)
                df1 = pd.concat(sup_dfs)
                df2 = pd.concat(qry_dfs)

            df1.to_csv('data/bats_{}{}{}{}{}_support.csv'.format(split, sufx1, sufx2, sufx3, sufx4))
            df2.to_csv('data/bats_{}{}{}{}{}_query.csv'.format(split, sufx1, sufx2, sufx3, sufx4))
            print('support split: {}, len: {}'.format(split, len(df1)))
            print('query split: {}, len: {}'.format(split, len(df2)))

        print('Done splitting for meta learning!')


    else:
        df_train = pd.concat(train)
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_test = pd.concat(test)
        df_dev = pd.concat(dev)
        #for m, word4 in enumerate(df_train['word4']):
        #    candidates = df_train['distractors'][m].copy()
        #    rand_id = np.random.randint(0, len(candidates)+1)
        #    candidates.insert(rand_id, word4)
        #    df_train['distractors'][m] = candidates
        #    df_train['label'][m] = rand_id

        #for index, row in df_train.iterrows():
        #    candidates = row['distractors'].copy()
        #    rand_id = np.random.randint(0, len(candidates))
        #    candidates.insert(rand_id, row['word4'])
        #    df_train.at[index, 'distractors'] = candidates
        #    df_train.loc[index, 'label'] = rand_id

        num_distractors = 0
        distractor_dfs = []

        #for i in range(num_distractors):
        #    replacements = df_quadruples['word4'].sample(len(df_quadruples)).tolist()
        #    distractors = df_quadruples.copy()
        #    distractors['word4'] = distractors['word4'].replace(distractors['word4'].tolist(), replacements)
        #    distractors['label'] = 0
        #    distractor_dfs.append(distractors)
        #if num_distractors > 0:
        #    df_distractors = pd.concat(distractor_dfs)
        #    df_quadruples = df_quadruples.append(df_distractors)
        sufx1 = '_strict'
        sufx2 = '_alts' if alts else ''
        sufx3 = '_distractors' if distractors else ''
        df_train.to_csv('data/bats_train{}{}{}.csv'.format(sufx1, sufx2, sufx3))
        df_dev.to_csv('data/bats_dev{}{}{}.csv'.format(sufx1, sufx2, sufx3))
        df_test.to_csv('data/bats_test{}{}{}.csv'.format(sufx1, sufx2, sufx3))

        print('Done splitting!')
        print(len(df_train), len(df_dev), len(df_test))


class BatsDataset(Dataset):

    def __init__(self, mode, strict=False, distractors=False, alts=False, meta_split=False, sup=True):
        self.distractors = distractors
        sufx = "_strict" if strict else ""
        sufx3 = "_distractors" if distractors else ""
        sufx2 = '_alts' if alts else ''
        sufx4 = '_meta' if meta_split else ''
        if meta_split:
            sufx5 = '_support' if sup else '_query'
        else:
            sufx5 = ''

        if mode == 'train':
            self.df = pd.read_csv('data/bats_train{}{}{}{}{}.csv'.format(sufx, sufx2, sufx3, sufx4, sufx5), index_col=0,
                                  converters={'alternatives': eval, 'distractors': eval})
        elif mode == 'dev':
            self.df = pd.read_csv('data/bats_dev{}{}{}{}{}.csv'.format(sufx, sufx2, sufx3, sufx4, sufx5), index_col=0,
                                  converters={'alternatives': eval, 'distractors': eval})
        elif mode == 'test':
            self.df = pd.read_csv('data/bats_test{}{}{}{}{}.csv'.format(sufx, sufx2, sufx3, sufx4, sufx5), index_col=0,
                                  converters={'alternatives': eval, 'distractors': eval})
        elif mode == 'all':
            if alts:
                self.df = pd.read_csv('data/bats_all_alts.csv', index_col=0)
            #                      converters={'alternatives': eval})
            else:
                self.df = pd.read_csv('data/bats_train{}{}.csv'.format(sufx, sufx2, sufx3), index_col=0,
                                      converters={'alternatives': eval, 'distractors': eval})
                df = pd.read_csv('data/bats_dev{}.csv'.format(sufx, sufx2), index_col=0,
                                 converters={'alternatives': eval, 'distractors': eval})
                self.df = self.df.append(df, ignore_index=True)
                df = pd.read_csv('data/bats_test{}.csv'.format(sufx, sufx2), index_col=0,
                                 converters={'alternatives': eval, 'distractors': eval})
                self.df = self.df.append(df, ignore_index=True)
        else:
            raise Exception("No valid mode. Choose 'train', 'dev' or 'test'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.distractors:
            data = self.df.iloc[idx, :3].values.tolist()
            correct_word = self.df.iloc[idx, 3]
            #label = self.df.iloc[idx, 5]
            #alternatives = self.df.iloc[idx, 7]
            candidates = self.df.iloc[idx, 6]
            label = self.df.iloc[idx, 5]
            relation_type = self.df.iloc[idx, -1]
            # source_id = self.df.iloc[idx, -1]
            # if len(alternatives) != 267:
            #    print(len(alternatives), idx)
            #sample = {'s1': data, 'candidates': candidates, 'label': label} #, 'alternatives': alternatives}
            #sample = (data, relation_type, label, candidates)
        else:
            data = self.df.iloc[idx, :3].values.tolist()
            label = self.df.iloc[idx, 3]
            alternatives = self.df.iloc[idx, 7]
            #source_id = self.df.iloc[idx, -1]
            #if len(alternatives) != 267:
            #    print(len(alternatives), idx)
            sample = {'s1': data, 'label': label, 'alternatives': alternatives}
        return sample

class BatsDatasetForLMs(Dataset):

    def __init__(self, mode, strict=False, distractors=False, alts=False):
        if strict:
            sufx = '_strict'
        else:
            sufx = ""
        if distractors:
            sufx2 = "_distractors"
        else:
            sufx2 = ""
        if mode == 'train':
            self.df = pd.read_csv('data/bats_train{}{}.csv'.format(sufx, sufx2), index_col=0,
                                    converters={'alternatives': eval})
        elif mode == 'dev':
            self.df = pd.read_csv('data/bats_dev{}{}.csv'.format(sufx, sufx2), index_col=0,
                                    converters={'alternatives': eval})
        elif mode == 'test':
            self.df = pd.read_csv('data/bats_test{}{}.csv'.format(sufx, sufx2), index_col=0,
                                    converters={'alternatives': eval})
        elif mode == 'all':
            if alts:
                self.df = pd.read_csv('data/bats_all_alts.csv', index_col=0)
            else:
                self.df = pd.read_csv('data/bats_train{}.csv'.format(sufx, sufx2), index_col=0,
                                        converters={'alternatives': eval})
                df = pd.read_csv('data/bats_dev{}.csv'.format(sufx, sufx2), index_col=0, converters={'alternatives': eval})
                self.df = self.df.append(df, ignore_index=True)
                df = pd.read_csv('data/bats_test{}.csv'.format(sufx, sufx2), index_col=0, converters={'alternatives': eval})
                self.df = self.df.append(df, ignore_index=True)
        else:
            raise Exception("No valid mode. Choose 'train', 'dev' or 'test'.")

    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rand_num = torch.randint(0, len(self.df), (1,)).item()
        s1 = self.df.iloc[rand_num, :4].values.tolist()
        s2 = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        alternatives = self.df.iloc[idx, 6]
        for i in range(10):
            if s2[2] in s1:
                try:
                    s1 = self.df.iloc[rand_num + i, :4].values.tolist()
                except Exception:
                    print('hit exception')
                    print('Value is in Example Sentence!!!!!')
                    print("value: {}, example: {}".format(s2[2], s1))
            else:
                break

        return {'s1': "If {} is like {}, then {} is like {}.".format(*s1),
                's2': "If {} is like {}, then {} is like".format(*s2),
                'label': label,
                'alternatives': alternatives}



class TurneyDataset(Dataset):
    def __init__(self):
        with open('data/turney.json', 'r', encoding='utf8') as f:
            data = json.load(f)
        word1 = []
        word2 = []
        topic1 = []
        topic2 = []
        for key in data:
            for first in data[key]:
                word1.append(first)
                word2.append(data[key][first])
                topic1.append(key.split('/')[0])
                topic2.append(key.split('/')[1])
        self.df = pd.DataFrame({
            'topic1': topic1,
            'topic2': topic2,
            'word1': word1,
            'word2': word2
        })
        print('done loading turney')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #sample = self.df.iloc[idx]
        #source, target = sample['topic'].split('/')
        #word = sample['word2']
        #label = sample['word1']
        #distractors = self.df['word2'].sample(4).tolist()
        #sample = [target, source, word]
        #tokenized_label = label
        # insert label in random position in the list of distractors
        #rand_idx = random.randint(0, 4)
        #distractors.insert(rand_idx, tokenized_label)
        sample = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        return {'s1': sample,
                'label': label, 'alternatives': ['None']}

def alternatives_helper(x):
    max_len = 3
    x = x.strip().split(',')
    x.extend(['None'] * (max_len - len(x)))
    return x

def split_scan_meta(seed=101):
    set_seeds(seed)
    with open('data/SCAN/SCAN_dataset.tsv', 'r', encoding='utf8') as f:
        df = pd.read_csv(f, sep='\t')
    print('loaded scan')
    fourth_words = df['src_word'].tolist()
    glove = Glove()
    glove.init_index()
    df['neighbors'] = glove.find_k_nearest_neighbors(fourth_words, 4)
    candidates_list = []
    label_list = []
    relations = []
    for id, row in df.iterrows():
        candidates = row['neighbors'].copy()
        rand_id = np.random.randint(0, len(candidates) + 1)
        label = rand_id
        candidates.insert(rand_id, row['src_word'])
        candidates_list.append(candidates)
        label_list.append(label)
        relations.append("{}-{}".format(row['target'], row['source']))
    df['distractors'] = candidates_list
    df['label'] = label_list
    df['subrelation'] = relations
    support = []
    query = []
    domains = list(zip(df['target'].tolist(), df['source'].tolist()))
    domains = list(dict.fromkeys(domains))
    for item in domains:
        dom_df = df.loc[(df['target'] == item[0]) & (df['source'] == item[1])]
        #print(len(dom_df))
        if len(dom_df) > 4:
            df_sup = dom_df.sample(4, random_state=seed)
            df_qry = dom_df[~dom_df.index.isin(df_sup.index)]
        else:
            df_sup = dom_df.sample(frac=0.5, random_state=seed)
            df_qry = dom_df[~dom_df.index.isin(df_sup.index)]
        support.append(df_sup)
        query.append(df_qry)
    query = pd.concat(query)
    support = pd.concat(support)
    query.to_csv('data/SCAN/scan_query.csv', sep="\t")
    support.to_csv('data/SCAN/scan_support.csv', sep="\t")
    return support, query



class ScanDataset(Dataset):
    def __init__(self, mode='standard'):
        if mode == 'support':
            with open('data/SCAN/scan_support.csv', 'r', encoding='utf8') as f:
                self.df = pd.read_csv(f, sep='\t', index_col=0, converters={'candidates': eval, 'distractors': eval})
            self.distractors = True
        elif mode == 'query':
            self.distractors = True
            with open('data/SCAN/scan_query.csv', 'r', encoding='utf8') as f:
                self.df = pd.read_csv(f, sep='\t', index_col=0, converters={'candidates': eval, 'distractors': eval})
        elif mode == 'standard':
            with open('data/SCAN/SCAN_dataset.tsv', 'r', encoding='utf8') as f:
                self.df = pd.read_csv(f, sep='\t', index_col=False)
            self.distractors = False
            # there were still two rows with same first three words but different fourth word. This should not
            # be the case since then the second instance should be part of the first's alternatives
            #first_doubled = self.df.loc[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='First')]
            last_doubled = self.df.loc[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='last')]
            self.df.drop(last_doubled.index, axis=0, inplace=True)
            for i, row in last_doubled.iterrows():
                first = self.df.loc[(self.df['source']==row['source']) & (self.df['target']==row['target']) & (self.df['targ_word']==row['targ_word'])].index
                self.df.at[first.values[0], 'alternatives'] = str(self.df.at[first.values[0], 'alternatives']) + ', '+ row['src_word']
            #with open('data/SCAN/SCAN_dataset_final.tsv', 'w', encoding='utf8') as f:
            #    self.df.to_csv(f, sep='\t', encoding='utf-8', index=False)
            #self.df.drop(last_doubled.index, axis=0, inplace=True)
        else:
            raise Exception("Invalid SCAN mode.")
        print('done loading SCAN')


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #sample = self.df.iloc[idx]
        #source, target = sample['topic'].split('/')
        #word = sample['word2']
        #label = sample['word1']
        #distractors = self.df['word2'].sample(4).tolist()
        #sample = [target, source, word]
        #tokenized_label = label
        if self.distractors:
            data = self.df.iloc[idx, :3].values.tolist()
            candidates = self.df.iloc[idx, 7]
            label = self.df.iloc[idx, 8]
            relation_type = self.df.iloc[idx, 9]
            sample = (data, relation_type, label, candidates, idx)
            return sample

        sample = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        alternatives = str(self.df.iloc[idx, 4])
        return {'s1': sample,
                'label': label,
                'alternatives': alternatives}


class ScanDatasetForLMs(Dataset):
    def __init__(self):
        with open('data/SCAN/SCAN_dataset.tsv', 'r', encoding='utf8') as f:
            self.df = pd.read_csv(f, sep='\t', index_col=False)
        self.distractors = False
        # there were still two rows with same first three words but different fourth word. This should not
        # be the case since then the second instance should be part of the first's alternatives
        # first_doubled = self.df.loc[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='First')]
        last_doubled = self.df.loc[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='last')]
        self.df.drop(last_doubled.index, axis=0, inplace=True)
        for i, row in last_doubled.iterrows():
            first = self.df.loc[(self.df['source'] == row['source']) & (self.df['target'] == row['target']) & (
                        self.df['targ_word'] == row['targ_word'])].index
            self.df.at[first.values[0], 'alternatives'] = str(self.df.at[first.values[0], 'alternatives']) + ', ' + row[
                'src_word']
        print('done loading SCAN')

    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rand_num = torch.randint(0, len(self.df), (1,)).item()
        s1 = self.df.iloc[rand_num, :4].values.tolist()
        s2 = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        alternatives = str(self.df.iloc[idx, 4])
        for i in range(10):
            if s2[2] in s1:
                try:
                    s1 = self.df.iloc[rand_num+i, :4].values.tolist()
                except Exception:
                    print('hit exception')
                    print('Value is in Example Sentence!!!!!')
                    print("value: {}, example: {}".format(s2[2], s1))
            else:
                break

        return {'s1': "If {} is like {}, then {} is like {}.".format(*s1),
                's2': "If {} is like {}, then {} is like".format(*s2),
                'label': label,
                'alternatives': alternatives}


class TurneyDatasetForLMs(Dataset):
    def __init__(self):
        with open('data/turney.json', 'r', encoding='utf8') as f:
            data = json.load(f)
        word1 =[]
        word2 = []
        topic1 = []
        topic2 = []
        for key in data:
            for first in data[key]:
                word1.append(first)
                word2.append(data[key][first])
                topic1.append(key.split('/')[0])
                topic2.append(key.split('/')[1])
        self.df = pd.DataFrame({
            'topic1': topic1,
            'topic2': topic2,
            'word1': word1,
            'word2': word2
        })
        print('done loading turney')

    def __len__(self):
        return int(len(self.df)/2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #sample = self.df.iloc[idx]
        #source, target = sample['topic'].split('/')
        #word = sample['word2']
        #label = sample['word1']
        # distractors = self.df['word2'].sample(4).tolist()
        #sample = [target, source, word]

        s1 = self.df.iloc[2*idx, :4].values.tolist()
        s2 = self.df.iloc[2*idx+1, :3].values.tolist()
        label = self.df.iloc[2*idx+1, 3]
        tokenized_label = label
        # insert label in random position in the list of distractors
        rand_idx = random.randint(0, 4)
        #distractors.insert(rand_idx, tokenized_label)
        flag = True
        for i in range(10):
            if s2[2] in s1:
                try:
                    print(i)
                    s1 = self.df.iloc[2*idx+i, :4].values.tolist()
                except Exception:
                    print('hit exception')
                    print('Value is in Example Sentence!!!!!')
                    print("value: {}, example: {}".format(s2[2], s1))
            else:
                flag = False
                break
        if flag:
            print(s1, s2)
        return {'s1': "If {} is like {}, then {} is like {}.".format(*s1),
                's2': "If {} is like {}, then {} is like".format(*s2),
                'label': tokenized_label,
                'alternatives': ['None']}

def eval_test_results_scan2(f_path):
    with open(f_path, 'rb') as f:
        results = pkl.load(f)
    #with open('results/answers.pkl', 'rb') as f:
    #    answers = pkl.load(f)
    with open('results/answers_df.csv', 'r', encoding='utf8') as f:
        filtered = pd.read_csv(f, sep='\t', index_col=0, converters={'candidates': eval, 'distractors': eval})
    seeds = [101, 202, 303, 404, 505, 606, 707]
    sc_accs = []
    met_accs = []
    gen_accs = []
    all_sc_res = []
    all_met_res = []
    answers_df = []
    flat_results = [item for sublist in results for item in sublist]
    filtered['results'] = flat_results
    sc = filtered.loc[filtered['analogy_type'] == 'science']
    met = filtered.loc[filtered['analogy_type'] == 'metaphor']
    sc_res = sc['results'].tolist()
    met_res = met['results'].tolist()
    sc_acc = sc['results'].mean()
    met_acc = met['results'].mean()

    print("GENERAL: {}, SCIENCE: {}, METAPHOR: {}".format(filtered['results'].mean(), sc_acc, met_acc))
    return sc_res, met_res

def eval_test_results_scan(f_path):
    with open(f_path, 'rb') as f:
        results = pkl.load(f)
    with open('results/answers.pkl', 'rb') as f:
        answers = pkl.load(f)
    seeds = [101, 202, 303, 404, 505, 606, 707]
    sc_accs = []
    met_accs = []
    gen_accs = []
    all_sc_res = []
    all_met_res = []
    answers_df = []
    for i, seed in enumerate(seeds):
        print(seed)
        split_scan_meta(seed=seed)
        #test_dataset_sup = ScanDataset(mode='support')
        test_dataset_qry = ScanDataset(mode="query")
        print(test_dataset_qry.df.head(5))
        filtered = test_dataset_qry.df.iloc[answers[i]]
        answers_df.append(filtered)
        filtered[seed] = results[i]
        sc = filtered.loc[filtered['analogy_type'] == 'science']
        met = filtered.loc[filtered['analogy_type'] == 'metaphor']
        sc_res = sc[seed].tolist()
        met_res = met[seed].tolist()
        all_sc_res.extend(sc_res)
        all_met_res.extend(met_res)
        sc_acc = sc[seed].mean()
        met_acc = met[seed].mean()
        sc_accs.append(sc_acc)
        met_accs.append(met_acc)
        gen_accs.append(filtered[seed].mean())
        print('done')
    answers_df = pd.concat(answers_df)
    answers_df.to_csv('results/answers_df.csv', sep='\t')
    print("GENERAL: {}, SCIENCE: {}, METAPHOR: {}".format(sum(gen_accs)/len(gen_accs), sum(sc_accs)/len(sc_accs), sum(met_accs)/len(met_accs)))
    return all_sc_res, all_met_res

def compare_models_scan(f_path):
    with open('results/answers.pkl', 'rb') as f:
        answers = pkl.load(f)
    seeds = [101, 202, 303, 404, 505, 606, 707]
    sc_accs = []
    met_accs = []
    gen_accs = []
    for i, seed in enumerate(seeds):
        print(seed)
        split_scan_meta(seed=seed)
        # test_dataset_sup = ScanDataset(mode='support')
        test_dataset_qry = ScanDataset(mode="query")
        filtered = test_dataset_qry.df.iloc[answers[i]]
        for p in f_path:
            with open('results/'+p+'.pkl', 'rb') as f:
                results = pkl.load(f)
            filtered[p] = results[i]
        for p in f_path:
            pos = filtered.loc[(filtered[p] == 1)]
            neg = filtered.loc[(filtered[p] == 0)]
            print('here')
        neg_all_but_one = filtered.loc[(filtered[f_path[1]] == 1) & (filtered[f_path[0]] == 0) & (filtered[f_path[3]] == 0) & (filtered[f_path[2]] == 0)]
        print('done')
