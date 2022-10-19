import torch
import json
from data.datasets import *
from models.models import *
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pickle as pkl
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from test_model import test_model, gpt2_tokenize, bert_tokenize, mult_bert_tokenize

def train_analogies(config_path):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    if config['parameters']['dataset'] == 'BATS':
        distractors = True
        strict = True
        #create_strict_bats_split(mode='strict', alts=False, distractors=distractors)
        train_dataset = BatsDataset('train', strict=strict, distractors=distractors)
        dev_dataset = BatsDataset('dev', strict=strict, distractors=distractors)
        test_dataset = BatsDataset('test', strict=strict, distractors=distractors)
        # dataset = BatsDataset('all')
        print(len(train_dataset), len(dev_dataset), len(test_dataset))
    elif config['parameters']['dataset'] == 'Turney':
        dataset = TurneyDataset()
    else:
        raise Exception("No valid dataset was given.")
    if config['parameters']['model'] == 'GloveWithGate':
        model = GloveWithGate(config).float().to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config['parameters']['model'] == 'BasicGlove':
        model = BasicGlove(config).float().to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config['parameters']['model'] == 'GloveSubtraction':
        model = BasicGlove(config).float().to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config['parameters']['model'] == 'GlovePredictWord':
        model = GlovePredictWord(config)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == "GloveHelensGate":
        model = GloveHelensGate(config).float().to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config['parameters']['model'] == "GloveWithVernasGate":
        model = GloveWithVernasGate(config).float().to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise Exception("No valid model given!")
    dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                            shuffle=True, num_workers=1)
    dataloader_dev = DataLoader(dev_dataset, batch_size=config['parameters']['batch_size'],
                                shuffle=True, num_workers=1)
    dataloader_test = DataLoader(test_dataset, batch_size=config['parameters']['batch_size'],
                                 shuffle=True, num_workers=1)

    optimizer = torch.optim.Adam(lr=config['parameters']['lr'], params=model.parameters())
    eval_every = 250
    max_acc = 0
    max_iteration = -1
    max_epoch = -1
    num_epochs = 4
    best_model = {}
    cnt = 0

    for epoch in range(num_epochs):
        for i_batch, batch in tqdm(enumerate(dataloader)):
            model.train()
            optimizer.zero_grad()
            #sample = batch['s1']
            label = batch['label'] # if distractors are used
            # if the prediction is over the entire vocabulary
            # label = torch.tensor([model.word2idx[batch['label'][i]] for i in range(len(batch['label']))])
            tokenized, label = tokenize(batch, model)
            tokenized = tokenized.to(device)
            label = label.to(device)
            prediction = model(tokenized)
            # result = torch.where(prediction > 0.5, 1, 0)
            loss = criterion(prediction.squeeze(), label.type_as(prediction))
            loss.backward()
            optimizer.step()

            if i_batch % eval_every == 0:

                model.eval()
                dev_results = []
                for i_dev_batch, dev_batch in enumerate(dataloader_dev):
                    with torch.no_grad():
                        #dev_sample = dev_batch['sample']
                        # if distractors are used
                        dev_label = dev_batch['label']
                        # if the prediction is over the entire vocabulary
                        # dev_label = torch.tensor([model.word2idx[dev_batch['label'][i]] for i in range(len(dev_batch['label']))])
                        dev_tokenized, dev_label = tokenize(dev_batch, model)
                        dev_tokenized = dev_tokenized.to(device)
                        dev_label = dev_label.to(device)
                        dev_prediction = model(dev_tokenized)
                        #dev_result = torch.argmax(dev_prediction, dim=1)
                        dev_result = torch.where(dev_prediction > 0.5, 1, 0)

                    found = torch.where(dev_result.squeeze().eq(dev_label), 1, 0)
                    #found = True if result == label else False
                    dev_results.extend(found.tolist())

                acc = dev_results.count(1) / len(dev_results)
                print('acc: {}'.format(acc))
                if acc > max_acc:
                    max_epoch = epoch
                    max_acc = acc
                    max_iteration = i_batch
                    best_model = model.get_state_dict()
                    torch.save(best_model, "trained_{}_{}.tar".format(config['parameters']['model'],
                                                                       config['parameters']['dataset']))
    print("MAX ACC: {}, @ ep: {} @it: {}".format(max_acc, max_epoch, max_iteration))
    model_params = torch.load("trained_{}_{}.tar".format(config['parameters']['model'],
                                                         config['parameters']['dataset']))
    model.load_model(model_params)

    model.eval()
    test_results = []
    for i_test_batch, test_batch in enumerate(dataloader_test):
        with torch.no_grad():
            test_label = test_batch['label']
            test_alternatives = test_batch['alternatives']
            # if the prediction is over the entire vocabulary
            # dev_label = torch.tensor([model.word2idx[dev_batch['label'][i]] for i in range(len(dev_batch['label']))])
            test_tokenized, _ = tokenize(test_batch, model)
            test_tokenized = test_tokenized.to(device)
            test_label = test_label.to(device)
            test_prediction = model(test_tokenized)
            # dev_result = torch.argmax(dev_prediction, dim=1)
            test_result = torch.where(test_prediction > 0.5, 1, 0)

        found = torch.where(test_result.squeeze().eq(test_label), 1, 0)
        #found = True if test_result == label else False
        test_results.extend(found.tolist())


    test_acc = test_results.count(1) / len(test_results)
    print('test acc: {}'.format(test_acc))
    return test_acc


def glove_tokenize(batch, model):
    words = list(zip(*batch['s1']))
    tokenized_batch = torch.ones((len(batch['s1']), len(words)), dtype=torch.int64)
    for b_id in range(len(words)):
        for w_id, w in enumerate(words[b_id]):
            tokenized_batch[w_id, b_id] = model.word2idx[w]
    labels = torch.tensor([model.word2idx[batch['label'][i]] for i in range(len(batch['label']))])
    return tokenized_batch, labels



def tokenize_bert(sample):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tmp = list(zip(*sample))
    words = []
    for i in range(len(tmp)):
        words.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                    tokenizer.mask_token))
    tokenized = tokenizer(words, padding=True, return_tensors='pt')
    return tokenized


def predict_analogies_bert(config_path):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    if config['parameters']['dataset'] == 'BATS':
        # create_strict_bats_split(None)
        train_dataset = BatsDataset('train', strict=True)
        dev_dataset = BatsDataset('dev', strict=True)
        test_dataset = BatsDataset('test', strict=True)
        # test_dataset = TurneyDataset()
        print(len(train_dataset), len(dev_dataset), len(test_dataset))
    elif config['parameters']['dataset'] == 'Turney':
        test_dataset = TurneyDataset()
    else:
        raise Exception("No valid dataset was given.")
    if config['parameters']['model'] == "BertForMaskedLM":
        model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        tokenize = bert_tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif config["parameters"]["model"] == "MultilingualBert":
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased').to(device)
        #tokenize = multilingual_bert_tokenize
        tokenize = mult_bert_tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    elif config['parameters']['model'] == "GPT2LMHeadModel":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id).to(device)
        #model.resize_token_embeddings(len(tokenizer))
        # model.requires_grad_(True)
        tokenize = gpt2_tokenize
    else:
        raise Exception("No valid model given!")
    dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                            shuffle=True, num_workers=0)
    dataloader_dev = DataLoader(dev_dataset, batch_size=config['parameters']['batch_size'],
                                shuffle=True, num_workers=1)
    dataloader_test = DataLoader(test_dataset, batch_size=config['parameters']['batch_size'], num_workers=1)
    optimizer = torch.optim.AdamW(lr=config['parameters']['lr'], params=model.parameters())

    eval_every = 1000
    max_acc = 0
    max_iteration = -1
    max_epoch = -1
    num_epochs = 2
    best_model = {}

    for epoch in range(num_epochs):
        for i_batch, batch in enumerate(dataloader):
            if i_batch % 100 == 0:
                print(i_batch)
            model.train()
            optimizer.zero_grad()
            tokenized, labels = tokenize(batch)
            tokenized = tokenized.to(device)
            labels = labels.to(device)
            output = model(**tokenized, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()


            if i_batch % eval_every == 0:
                print('entering validation mode')
                acc = test_model(model, dev_dataset, tokenize, tokenizer, config['parameters']['model'])
                print('dev acc: {}'.format(acc))
                if acc > max_acc:
                    max_epoch = epoch
                    max_acc = acc
                    max_iteration = i_batch
                    best_model = model.state_dict()
                    torch.save(best_model, "saved_models/trained_{}_{}.tar".format(config['parameters']['model'],
                                                                      config['parameters']['dataset']))

    print("MAX ACC: {}, @ ep: {} @it: {}".format(max_acc, max_epoch, max_iteration))

    model_params = torch.load("saved_models/trained_{}_{}.tar".format(config['parameters']['model'],
                                                         config['parameters']['dataset']))
    model.load_state_dict(model_params)

    test_acc = test_model(model, test_dataset, tokenize, tokenizer, config['parameters']['model'])
    print('test acc: {}'.format(test_acc))
    return test_acc


def glove_tokenize_turney(batch, model):
    words = list(zip(*batch['s1']))
    # tokenized_batch = torch.ones((len(batch['s1']), len(words)), dtype=torch.int64)
    tokenized_batch = []
    for b_id in range(len(words)):
        per_batch = []
        for w_id, w in enumerate(words[b_id]):
            if len(w.split(' ')) > 1:
                tmp = []
                for word in w.split(' '):
                    tmp.append(model.word2idx[word])
                per_batch.append(tmp)
            else:
                per_batch.append([model.word2idx[w]])
        tokenized_batch.append(per_batch)
    labels = torch.tensor([model.word2idx[batch['label'][i]] for i in range(len(batch['label']))])
    return tokenized_batch, labels

def predict_analogies_glove(config_path):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    if config['parameters']['dataset'] == 'BATS':
        #create_strict_bats_split()
        train_dataset = BatsDataset('train', strict=True, distractors=True)
        dev_dataset = BatsDataset('dev', strict=True, distractors=True)
        test_dataset = BatsDataset('test', strict=True, distractors=True)
        print(len(train_dataset), len(dev_dataset), len(test_dataset))
    elif config['parameters']['dataset'] == 'Turney':
        dataset = TurneyDataset()
    else:
        raise Exception("No valid dataset was given.")
    if config['parameters']['model'] == 'GlovePredictWordSubtraction':
        model = GlovePredictWordSubtraction(config).to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == 'GlovePredictWord':
        model = GlovePredictWord(config).to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == 'GloveNoTraining':
        model = GloveNoTraining(config).to(device)
        tokenize = glove_tokenize_turney
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == 'GlovePredictWordWithGate':
        model = GlovePredictWordWithGate(config).to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == "GlovePredictWordWithAttention":
        model = GlovePredictWordWithAttention(config).to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == 'GlovePredictWordWithHelensGate':
        model = GlovePredictWordWithHelensGate(config).to(device)
        tokenize = glove_tokenize
        criterion = torch.nn.CrossEntropyLoss()
    elif config['parameters']['model'] == "BasicBert":
        # model = BasicBert(config).to(device)
        model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        model.requires_grad_(True)
        tokenize = bert_tokenize
        criterion = torch.nn.CrossEntropyLoss()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise Exception("No valid model given!")

    optimizer = torch.optim.Adam(lr=config['parameters']['lr'], params=model.parameters())
    eval_every = 250
    num_epochs = 2
    max_epoch = -1
    max_acc = -1
    max_iteration = -1
    best_model = None

    for epoch in range(num_epochs):
        dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'],
                                shuffle=True, num_workers=3)
        dataloader_dev = DataLoader(dev_dataset, batch_size=16,
                                    shuffle=True, num_workers=1)
        for i_batch, batch in tqdm(enumerate(dataloader)):
            model.train()
            optimizer.zero_grad()
            sample = batch['s1']
            #label = tokenize(batch['label']).to(device) # if distractors are used
            # if the prediction is over the entire vocabulary

            # labels = batch['label']
            alternatives = list(zip(*batch['alternatives']))
            tokenized, labels = tokenize(batch, model)
            labels = labels
            output = model(tokenized)
            k = 1
            xq = output.detach().numpy().astype(dtype='float32')
            D, I = model.index.search(xq, k)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i_batch % eval_every == 0:
                model.eval()
                dev_results = []
                for i_dev_batch, dev_batch in enumerate(dataloader_dev):
                    dev_labels = dev_batch['label']
                    dev_alternatives = list(zip(*dev_batch['alternatives']))
                    dev_tokenized, _ = tokenize(dev_batch, model)
                    with torch.no_grad():
                        dev_output = model(dev_tokenized)
                    k = 10
                    xq = dev_output.numpy().astype(dtype='float32')
                    D, I = model.index.search(xq, k)
                    for i, row in enumerate(I):
                        found = False
                        for idx in row:
                            if model.idx2word[idx] in dev_labels[i].split(' ') or model.idx2word[idx] in dev_alternatives[i]:
                                found = True
                        dev_results.append(found)
                    acc = sum(dev_results)/len(dev_results)
                    print('dev acc: {}'.format(acc))
                    if acc > max_acc:
                        max_epoch = epoch
                        max_acc = acc
                        max_iteration = i_batch
                        best_model = model.state_dict()
                        torch.save(best_model, "trained_{}_{}.tar".format(config['parameters']['model'],
                                                                          config['parameters']['dataset']))



                print("MAX ACC: {}, @ ep: {} @it: {}".format(max_acc, max_epoch, max_iteration))


if __name__ == "__main__":
    k = 1
    accs = []
    for i in range(k):
        acc = predict_analogies_bert('config.json')
        accs.append(acc)
    print("Averaged Accs: {}".format(sum(accs)/len(accs)))
