import torch
import json
from data.datasets import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pickle as pkl
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from argparse import ArgumentParser
import io
#from predict_word_analogy import evaluate_outputs

def bert_tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    #labels = tokenizer(batch['label'], add_special_tokens=False, return_tensors='pt', padding='max_length',
    #                   max_length=10)['input_ids']
    vocab = tokenizer.get_vocab()
    labels = torch.tensor([vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                           for word in batch['label']])
    #only_labels = tokenizer(labels, add_special_tokens=False)
    if isinstance(batch['s1'][0], tuple):
        tmp = list(zip(*batch['s1']))
        words = []
        tmp_labels = []
        for i in range(len(tmp)):
            # words.append(list(tp)+[tokenizer.mask_token])
            #masked_tokens = ' '.join([tokenizer.mask_token] * len(only_labels['input_ids'][i]))
            words.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                        tokenizer.mask_token))
            #words.append('{} has the same relationship to {} as {} to {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
            #                                                            tokenizer.mask_token))
            # tmp_labels.append(list(tp)+[labels[i]])
            # only give first token as label
            tmp_labels.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                        labels[i]))
            #tmp_labels.append('{} has the same relationship to {} as {} to {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
            #                                                                 labels[i]))
        tokenized_batch = tokenizer(words, padding='max_length', return_tensors="pt", max_length=50, truncation=True)
        tokenized_labels = tokenized_batch['input_ids'].clone()
        batch_mask_index, mask_index = torch.where(tokenized_labels==tokenizer.mask_token_id)
        tokenized_labels[batch_mask_index, mask_index] = labels
    return tokenized_batch, tokenized_labels

def tokenize_bert_helper(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    words = []
    sample1 = batch['s1']
    sample2 = batch['s2']
    vocab = tokenizer.get_vocab()
    labels = torch.tensor([vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                           for word in batch['label']])
    #only_labels = tokenizer(labels, add_special_tokens=False)
    tok_labels = []
    for i in range(len(sample1)):
        #masked_tokens = ' '.join([tokenizer.mask_token] * len(only_labels['input_ids'][i]))
        words.append('{} {} {}.'.format(sample1[i], sample2[i], tokenizer.mask_token))
        tok_labels.append('{} {} {}.'.format(sample1[i], sample2[i], labels[i]))
    tokenized = tokenizer(words, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    tokenized_labels = tokenizer(tok_labels, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    return tokenized, tokenized_labels

def mult_bert_tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    vocab = tokenizer.get_vocab()
    labels = torch.tensor([vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                           for word in batch['label']])
    #only_labels = tokenizer(labels, add_special_tokens=False)
    if isinstance(batch['s1'][0], tuple):
        tmp = list(zip(*batch['s1']))
        words = []
        tmp_labels = []
        for i in range(len(tmp)):
            # words.append(list(tp)+[tokenizer.mask_token])
            #masked_tokens = ' '.join([tokenizer.mask_token] * len(only_labels['input_ids'][i]))
            words.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                        tokenizer.mask_token))
            #words.append('{} has the same relationship to {} as {} to {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
            #                                                            tokenizer.mask_token))
            # tmp_labels.append(list(tp)+[labels[i]])
            tmp_labels.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                        labels[i]))
            #tmp_labels.append('{} has the same relationship to {} as {} to {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
            #                                                                 labels[i]))
        tokenized_batch = tokenizer(words, padding='max_length', return_tensors="pt", max_length=50, truncation=True)
        tokenized_labels = tokenized_batch['input_ids'].clone()
        batch_mask_index, mask_index = torch.where(tokenized_labels == tokenizer.mask_token_id)
        tokenized_labels[batch_mask_index, mask_index] = labels
    return tokenized_batch, tokenized_labels

def tokenize_mult_bert_helper(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    words = []
    sample1 = batch['s1']
    sample2 = batch['s2']
    vocab = tokenizer.get_vocab()
    labels = torch.tensor([vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                           for word in batch['label']])
    #only_labels = tokenizer(labels, add_special_tokens=False)
    tok_labels = []
    for i in range(len(sample1)):
        #masked_tokens = ' '.join([tokenizer.mask_token] * len(only_labels['input_ids'][i]))
        words.append('{} {} {}.'.format(sample1[i], sample2[i], tokenizer.mask_token))
        tok_labels.append('{} {} {}.'.format(sample1[i], sample2[i], labels[i]))
    tokenized = tokenizer(words, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    tokenized_labels = tokenizer(tok_labels, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    return tokenized, tokenized_labels

def gpt2_tokenize(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    #vocab = {(w if w[0] != 'Ġ' else w[1:]): val for (w, val) in vocab.items()}
    labels = torch.tensor(
        [vocab['Ġ' + word] if ('Ġ' + word) in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
         for word in batch['label']])
    if isinstance(batch['s1'][0], tuple):
        tmp = list(zip(*batch['s1']))
        words = []
        tmp_labels = []
        for i in range(len(tmp)):
            # words.append(list(tp)+[tokenizer.mask_token])
            words.append('If {} is like {}, then {} is like'.format(tmp[i][0], tmp[i][1], tmp[i][2]))
            #words.append('{} {} {} '.format(tmp[i][0], tmp[i][1], tmp[i][2]))
            # tmp_labels.append(list(tp)+[labels[i]])
            tmp_labels.append('If {} is like {}, then {} is like {}'.format(tmp[i][0], tmp[i][1], tmp[i][2],
                                                                            labels[i]))
            #tmp_labels.append('{} {} {} {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2],
            #                                                                 labels[i]))
        tokenized_batch = tokenizer(words, padding='max_length', return_tensors="pt", max_length=50, truncation=True)
        tokenized_labels = tokenized_batch['input_ids'].clone()
        mask_index = torch.sum(torch.tensor(tokenized_labels) != tokenizer.eos_token_id, dim=1)
        batch_mask_index = torch.arange(0, len(tokenized_labels), dtype=torch.int64)
        tokenized_labels[batch_mask_index, mask_index] = labels
    return tokenized_batch, tokenized_labels

def tokenize_gpt2_helper(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    # vocab = {(w if w[0] != 'Ġ' else w[1:]): val for (w, val) in vocab.items()}
    labels = torch.tensor(
        [vocab['Ġ' + word] if ('Ġ' + word) in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
         for word in batch['label']])
    sample1 = batch['s1']
    sample2 = batch['s2']
    words = []
    tok_labels = []
    for i in range(len(sample1)):
        words.append('{} {}'.format(sample1[i], sample2[i]))
        tok_labels.append('{} {} {}'.format(sample1[i], sample2[i], labels[i]))
    tokenized = tokenizer(words, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    tokenized_labels = tokenizer(tok_labels, padding='max_length', return_tensors='pt', max_length=50, truncation=True)
    return tokenized, tokenized_labels

def check_tokens(dataset, tokenizer):
    results = []
    vocab = tokenizer.get_vocab()
    if type(tokenizer).__name__ == 'GPT2Tokenizer':
        vocab = [w if w[0] != 'Ġ' else w[1:] for w in vocab]
    predict_words = dataset.df.iloc[:, 3].values.tolist()
    alternatives = dataset.df['alternatives'].values.tolist()
    for i, word in enumerate(predict_words):
        found = False
        alts = alternatives[i]
        # if target consists of multiple words
        if len(word.split(' ')) > 1:
            words = set(word.split(' '))
            if not words.intersection(set(alts)):
                found = False
            else:
                found = True
                results.append(found)
                continue
        if word in vocab:
            found = True
            results.append(found)
            continue
        for alt in alts:
            if alt == 'None':
                break
            if alt in vocab:
                found = True
                break
        results.append(found)
    res = sum(results)/len(results)
    print(res)

def calc_accs_per_type(results, dataset):
    assert len(results) == len(dataset)
    df = dataset.df
    df['res'] = results
    science = df.loc[df['analogy_type'] == 'science']
    meta = df.loc[df['analogy_type'] == 'metaphor']
    science_acc = science['res'].mean()
    meta_acc = meta['res'].mean()
    print('SCIENCE: {}, METAPHOR: {}'.format(science_acc, meta_acc))


def prepare_alternatives_bert(tokenizer, alternatives, device):
    alt_dict = {}
    max_len_alts = 0
    nan_token_id = tokenizer("nan", add_special_tokens=False)['input_ids'][0]
    for g, alt_list in enumerate(alternatives):
        if type(alt_list).__name__ == 'str':
            split_alternatives = alt_list.split(", ")
        else:
            split_alternatives = alt_list
        if len(split_alternatives) > max_len_alts:
            max_len_alts = len(split_alternatives)
        alt_dict[g] = []
        if (split_alternatives == ["nan"]) or (all("None" == item for item in split_alternatives)):
            continue
        for word in split_alternatives:
            if word == '' or "None":
                continue
            else:
                alt_dict[g].append(tokenizer(word, add_special_tokens=False)['input_ids'][0])
    tokenized_alternatives = torch.zeros((len(alternatives), max_len_alts)).to(device)
    tokenized_alternatives = tokenized_alternatives + nan_token_id
    for idx in alt_dict:
        for j, token_id in enumerate(alt_dict[idx]):
            tokenized_alternatives[idx, j] = token_id

    return tokenized_alternatives

def prepare_alternatives_gpt(tokenizer, alternatives, device):
    alt_dict = {}
    max_len_alts = 0
    nan_token_id = tokenizer("nan", add_special_tokens=False)['input_ids'][0]
    for g, alt_list in enumerate(alternatives):
        if type(alt_list).__name__ == 'str':
            split_alternatives = alt_list.split(", ")
        else:
            split_alternatives = alt_list
        if len(split_alternatives) > max_len_alts:
            max_len_alts = len(split_alternatives)
        alt_dict[g] = []
        if (split_alternatives == ["nan"]) or (all("None" == item for item in split_alternatives)):
            continue
        for word in split_alternatives:
            if word == '' or "None":
                continue
            else:
                alt_dict[g].append(tokenizer(word, add_special_tokens=False)['input_ids'][0])
    tokenized_alternatives = torch.zeros((len(alternatives), max_len_alts)).to(device)
    tokenized_alternatives = tokenized_alternatives + nan_token_id
    for idx in alt_dict:
        for j, token_id in enumerate(alt_dict[idx]):
            tokenized_alternatives[idx, j] = token_id
    return tokenized_alternatives



class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def test_model(model, dataset, tokenize, tokenizer, model_name, helper_sent=False, batch_size=4):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    decoded_outputs = []
    outputs = []
    all_labels = []
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_results = []

    print(model_name, type(dataset).__name__, len(dataset), flush=True)
    fname = 'saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model_name, type(dataset).__name__, helper_sent)
    if os.path.isfile(fname):
        print("loading from: ", fname)
        with open(fname, 'rb') as f:
            if device == "cpu":
                outputs = CPU_Unpickler(f).load()
            else:
                outputs = pkl.load(f)
            #outputs = outputs.to(device)
    else:
        for i_batch, batch in tqdm(enumerate(dataloader)):
            sample1 = batch['s1']
            vocab = tokenizer.get_vocab()
            # only take the first token of the label wor
            if 'GPT' in type(model).__name__:
            #    vocab = {(w if w[0] != 'Ġ' else w[1:]): val for (w, val) in vocab.items()}
                labels = torch.tensor(
                    [vocab['Ġ'+word] if ('Ġ'+word) in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                     for word in batch['label']]).to(device)
            else:
                labels = torch.tensor(
                    [vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
                     for word in batch['label']]).to(device)

            alternatives = batch['alternatives'] #list(zip(*batch['alternatives']))
            if 'Bert' in model_name:
                alternatives = prepare_alternatives_bert(tokenizer, alternatives, device)
            elif 'GPT' in model_name:
                alternatives = prepare_alternatives_gpt(tokenizer, alternatives, device)

            if helper_sent:
                sample2 = batch['s2']
                tokenized, _ = tokenize(batch)
                tokenized = tokenized.to(device)
            else:
                tokenized, _ = tokenize(batch)
                tokenized = tokenized.to(device)

            if 'Bert' in model_name:
                mask_index_batch, mask_index = torch.where(torch.tensor(tokenized['input_ids']) == tokenizer.mask_token_id)
            elif "GPT" in model_name:
                mask_index = torch.sum(torch.tensor(tokenized['input_ids']) != tokenizer.eos_token_id, dim=1) - 1
                mask_index_batch = torch.arange(0, len(tokenized['input_ids']), dtype=torch.int64).to(device)
            with torch.no_grad():
                output = model(**tokenized)
                output = output.logits
                output = F.softmax(output, dim=-1)
            mask_word = output[mask_index_batch, mask_index, :]
            _, top_k = torch.topk(mask_word, 10, dim=-1)
            for k, entry in enumerate(top_k):
                all_labels.append(labels[k].item())
                #words = tokenizer.decode(entry)
                words = tokenizer.decode(entry)
                found = False
                decoded_outputs.append(words.split(' '))
                outputs.append(entry)
                found = False
                for idx in entry:
                    if labels[k] == idx or idx in alternatives[k]:
                        found = True
                if found == False and batch['label'][k] in tokenizer.decode(entry).split(' '):
                    print('should not reach this')
                test_results.append(found)
            output = None
        acc = test_results.count(1) / len(test_results)
        print('acc: {}'.format(acc), flush=True)
        if not 'Bats' in type(dataset).__name__:
            calc_accs_per_type(test_results, dataset)
        with open('saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model_name, type(dataset).__name__, helper_sent), 'wb') as f:
            pkl.dump(outputs, f)
    acc, mrr, recall10 = evaluate_outputs(outputs, dataset, tokenizer, model_name, device)
    return acc

def run_test(config_path):
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    if config['parameters']['dataset'] == 'BATS':
        #create_strict_bats_split('all', alts=True)
        if config['parameters']['helper_sent']:
            dataset = BatsDatasetForLMs('all')
            #dataset.df = dataset.df.dropna()
        else:
            #dataset = BatsDataset('all')
            #dataset.df = dataset.df.dropna()
            dataset = BatsDataset(mode='test', strict=True)
    elif config['parameters']['dataset'] == 'Turney':
        if config['parameters']['helper_sent']:
            dataset = TurneyDatasetForLMs()
        else:
            dataset = TurneyDataset()
    elif config['parameters']['dataset'] == 'Scan':
        if config['parameters']['helper_sent']:
            dataset = ScanDatasetForLMs()
        else:
            dataset = ScanDataset(mode="standard")
    else:
        raise Exception("No valid dataset was given.")
    if config['parameters']['model'] == "BertForMaskedLM":
        # model = BasicBert(config).to(device)
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        for param in model.parameters():
            param.requires_grad = False
        if config['parameters']['helper_sent']:
            tokenize = tokenize_bert_helper
        else:
            tokenize = bert_tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif config["parameters"]["model"] == "MultilingualBert":
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        for param in model.parameters():
            param.requires_grad = False
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if config['parameters']['helper_sent']:
            tokenize = tokenize_mult_bert_helper
        else:
            tokenize = mult_bert_tokenize
    elif config['parameters']['model'] == "GPT2LMHeadModel":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
        for param in model.parameters():
            param.requires_grad = False
        if config['parameters']['helper_sent']:
            tokenize = tokenize_gpt2_helper
        else:
            tokenize = gpt2_tokenize
    else:
        raise Exception("No valid model given!")

    #model_params = torch.load("trained_BasicBert_BATS.tar")
    #model_params = torch.load("saved_models/trained_{}_{}.tar".format(config['parameters']['model'],
    #                                                                  'BATS'))
    #model.load_state_dict(model_params)
    #model_params = None
    acc = test_model(model, dataset, tokenize, tokenizer, helper_sent=config['parameters']['helper_sent'],
                     batch_size=config['parameters']['batch_size'], model_name=config['parameters']['model'])
    return acc

def evaluate_saved_test(model_name, dataset, helper_sent):
    with open('saved_tests/eval_{}_{}_untrained_{}.pkl'.format(model_name, type(dataset).__name__, helper_sent),
              'wb') as f:
        test_results = pkl.load(f)
    acc, mrr, recall10 = evaluate_outputs(outputs, all_labels, dataset, tokenizer, model_name, device)


def evaluate_outputs(outputs, dataset, tokenizer, model_name, device):
    df = dataset.df
    #df['helper_row'] = outputs.keys()
    df['outputs'] = outputs
    labels = list(df['src_word']) if (isinstance(dataset, ScanDataset) or (isinstance(dataset, ScanDatasetForLMs))) else list(df['word4'])
    #labels = [tokenizer(item, add_special_tokens=False)['input_ids'][0] for item in labels]
    if 'GPT' in type(tokenizer).__name__:
        #    vocab = {(w if w[0] != 'Ġ' else w[1:]): val for (w, val) in vocab.items()}
        vocab = tokenizer.get_vocab()
        labels = torch.tensor(
            [vocab['Ġ' + word] if ('Ġ' + word) in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
             for word in labels]).to(device)
    else:
        vocab = tokenizer.get_vocab()
        labels = torch.tensor(
            [vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
             for word in labels]).to(device)
    df['tok_labels'] = labels.cpu()
    if "Scan" in type(dataset).__name__:
        alternatives = [str(item) for item in list(df['alternatives'])]
    else:
        alternatives = list(df['alternatives'])
    if 'Bert' in model_name:
        alternatives = prepare_alternatives_bert(tokenizer, alternatives, device)
    elif 'GPT' in model_name:
        alternatives = prepare_alternatives_gpt(tokenizer, alternatives, device)
    df['tok_alternatives'] = alternatives.cpu()


    mrr, _ = mean_reciprocal_rank(outputs, labels, alternatives, tokenizer)
    recall10, _ = recall_at_k(outputs, labels, alternatives, tokenizer, 10)
    recall5, _ = recall_at_k(outputs, labels, alternatives, tokenizer, 5)
    accuracy, _ = calc_accuracy(outputs, labels, alternatives, tokenizer)
    accuracy10, _ = calc_accuracy(outputs, labels, alternatives, tokenizer, k=10)
    print("Acc@10: {}, General: Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(np.format_float_positional(accuracy10, precision=3), accuracy, mrr, recall10, recall5))

    science_outputs = list(df.loc[df['analogy_type'] == 'science']['outputs'])
    science_labels = list(df.loc[df['analogy_type'] == 'science']['tok_labels'])
    science_alternatives = [torch.tensor(item) for item in list(df.loc[df['analogy_type'] == 'science']['tok_alternatives'])]
    science_mrr, _ = mean_reciprocal_rank(science_outputs, science_labels, science_alternatives, tokenizer)
    science_recall10, _ = recall_at_k(science_outputs, science_labels, science_alternatives, tokenizer, 10)
    science_recall5, _ = recall_at_k(science_outputs, science_labels, science_alternatives, tokenizer, 5)
    science_accuracy, _ = calc_accuracy(science_outputs, science_labels, science_alternatives, tokenizer)
    science_accuracy10, _ = calc_accuracy(science_outputs, science_labels, science_alternatives, tokenizer, k=10)

    metaphor_outputs = list(df.loc[df['analogy_type'] == 'metaphor']['outputs'])
    metaphor_labels = list(df.loc[df['analogy_type'] == 'metaphor']['tok_labels'])
    metaphor_alternatives = [torch.tensor(item) for item in list(df.loc[df['analogy_type'] == 'metaphor']['tok_alternatives'])]
    metaphor_mrr, _ = mean_reciprocal_rank(metaphor_outputs, metaphor_labels, metaphor_alternatives, tokenizer)
    metaphor_recall10, _ = recall_at_k(metaphor_outputs, metaphor_labels, metaphor_alternatives, tokenizer, 10)
    metaphor_recall5, _ = recall_at_k(metaphor_outputs, metaphor_labels, metaphor_alternatives, tokenizer, 5)
    metaphor_accuracy, _ = calc_accuracy(metaphor_outputs, metaphor_labels, metaphor_alternatives, tokenizer)
    metaphor_accuracy10, _ = calc_accuracy(metaphor_outputs, metaphor_labels, metaphor_alternatives, tokenizer, k=10)


    print("Science: Accuracy10: {}, Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(science_accuracy10, science_accuracy, science_mrr, science_recall10, science_recall5))
    print("Metaphor: Accuracy10: {}, Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(metaphor_accuracy10, metaphor_accuracy, metaphor_mrr, metaphor_recall10, metaphor_recall5))
    print("\n")
    print("\n")
    print("&{} &{} &{} &{} &{} &{} &{} &{} &{} &{} &{} &{}".format(np.format_float_positional(accuracy, precision=3),
                                   np.format_float_positional(mrr, precision=3),
                                   np.format_float_positional(recall10, precision=3),
                                   np.format_float_positional(recall5, precision=3),
                                    np.format_float_positional(science_accuracy, precision=3),
                                    np.format_float_positional(science_mrr, precision=3),
                                    np.format_float_positional(science_recall10, precision=3),
                                    np.format_float_positional(science_recall5, precision=3),
                                    np.format_float_positional(metaphor_accuracy, precision=3),
                                    np.format_float_positional(metaphor_mrr, precision=3),
                                    np.format_float_positional(metaphor_recall10, precision=3),
                                    np.format_float_positional(metaphor_recall5, precision=3)
    ))
    return accuracy, mrr, recall10

def mean_reciprocal_rank(outputs, labels, alternatives, tokenizer):
    nan_token = tokenizer('nan', add_special_tokens=False)['input_ids'][0]
    rec_ranks = []
    for i, sample in enumerate(outputs):
        found = False
        for j, idx in enumerate(sample):
            if (labels[i] == idx or idx in alternatives[i]) and (idx != nan_token):
                rec_ranks.append(1 / (j + 1))
                found = True
                break
        if not found:
            rec_ranks.append(0)
    mrr = np.mean(rec_ranks)
    return mrr, rec_ranks


def calc_accuracy(outputs, labels, alternatives, tokenizer, k=1):
    predictions = []
    nan_token = tokenizer('nan', add_special_tokens=False)['input_ids'][0]
    for i, sample in enumerate(outputs):
        found = False
        for j, idx in enumerate(sample):
            if j >= k:
                break
            if (labels[i] == idx or idx in alternatives[i]) and (idx != nan_token):
                predictions.append(1)
                found = True
                break
        if not found:
            predictions.append(0)
    acc = np.mean(predictions)
    return acc, predictions

def recall_at_k(outputs, labels, alternatives, tokenizer, k=10):
    """
    Note that outputs needs to include at least k predictions per data sample
    :param outputs:
    :param labels:
    :param alternatives:
    :param k:
    :return: the recall at k value
    """
    recalls = []
    for i, sample in enumerate(labels):
        # take nan token out of alternatives
        nan_token = tokenizer('nan', add_special_tokens=False)['input_ids'][0]
        relevant = alternatives[i]
        relevant = torch.cat((relevant, torch.tensor(labels[i]).unsqueeze(0)), 0)
        relevant = relevant[torch.where(relevant!=nan_token)]
        found = 0
        #relevant.append(labels[i])
        #relevant = set(relevant) - {'nan'}
        for word in relevant:
            if word in outputs[i][:k]:
                found += 1
        recalls.append(found/len(relevant))
    avg_recall_at_k = np.mean(recalls)
    return avg_recall_at_k, recalls




def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='test_config.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.config)
    accs = []
    num_iterations = 1
    for i in range(num_iterations):
        accs.append(run_test(args.config))
    print('final acc: {}'.format(sum(accs) / num_iterations))
