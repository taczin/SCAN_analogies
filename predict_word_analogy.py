import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed
from torch.nn import functional
import json
from data.datasets import *
from tqdm import tqdm
from models.models import *
from torch.utils.data import DataLoader
#from train_analogy import glove_tokenize
import pickle as pkl


def calc_accs_per_type(results, dataset):
    assert len(results) == len(dataset)
    df = dataset.df
    df['res'] = results
    science = df.loc[df['analogy_type'] == 'science']
    meta = df.loc[df['analogy_type'] == 'metaphor']
    science_acc = science['res'].mean()
    meta_acc = meta['res'].mean()
    print('SCIENCE: {}, METAPHOR: {}'.format(science_acc, meta_acc))

def tokenize_bert(sample):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tmp = list(zip(*sample))
    #tokenized_label = tokenizer(label, add_special_tokens=False)
    words = []
    for i in range(len(tmp)):
        mask_tokens = [tokenizer.mask_token] #* len(tokenized_label['input_ids'][i])
        words.append('If {} is like {}, then {} is like {}.'.format(tmp[i][0], tmp[i][1], tmp[i][2], mask_tokens))
    tokenized = tokenizer(words, padding=True, return_tensors='pt')
    return tokenized

def tokenize_gpt2(sample):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tmp = list(zip(*sample))
    words = []
    for i in range(len(tmp)):
        words.append('If {} is like {}, then {} is like'.format(tmp[i][0], tmp[i][1], tmp[i][2]))
    tokenized = tokenizer(words, padding=True, return_tensors='pt')
    return tokenized

def tokenize_bert_helper(sample1, sample2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    words = []
    for i in range(len(sample1)):
        words.append('{} {} {}.'.format(sample1[i], sample2[i], tokenizer.mask_token))
    tokenized = tokenizer(words, padding=True, return_tensors='pt')
    return tokenized

def tokenize_gpt2_helper(sample1, sample2):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    words = []
    for i in range(len(sample1)):
        words.append('{} {}'.format(sample1[i], sample2[i]))
    tokenized = tokenizer(words, padding=True, return_tensors='pt')
    return tokenized

def beam_search_generation(sentence, label, k, device=device):
    generator = pipeline('text-generation', model='gpt2', device=device)
    set_seed(42)
    generator("Hello, I'm a language model,", max_length=30, num_return_sequences=k)


def solve_analogy(config_path):
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    if config["parameters"]["model"] == "BertForMaskedLM":
        model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if config['parameters']['helper_sent']:
            tokenize = tokenize_bert_helper
        else:
            tokenize = tokenize_bert
    elif config["parameters"]["model"] == "MultilingualBert":
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', return_dict=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if config['parameters']['helper_sent']:
            tokenize = tokenize_bert_helper
        else:
            tokenize = tokenize_bert
    elif config["parameters"]["model"] == "GPT2LMHeadModel":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id).to(device)
        # model.resize_token_embeddings(len(tokenizer))
        for param in model.parameters():
            param.requires_grad = False

        if config['parameters']['helper_sent']:
            tokenize = tokenize_gpt2_helper
        else:
            tokenize = tokenize_gpt2
    elif config["parameters"]["model"] == "BertWithGate":
        model = BertWithGate(config).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenize = tokenize_bert
    else:
        raise Exception("Could not identify model.")
    if config['parameters']['dataset'] == 'BATS':
        # create_bats_split(None)
        #train_dataset = BatsDataset('train')
        #dev_dataset = BatsDataset('dev')
        # dataset = BatsDataset('test')
        #print(len(train_dataset), len(dev_dataset), len(dataset))
        if config['parameters']['helper_sent']:
            dataset = BatsDatasetForLMs('all')
        else:
            dataset = BatsDataset('all')
        #dataset = BatsDataset('dev', strict=True)
        print(len(dataset))
    elif config['parameters']['dataset'] == 'Turney':
        if config['parameters']['helper_sent']:
            dataset = TurneyDatasetForLMs()
        else:
            dataset = TurneyDataset()
    elif config['parameters']['dataset'] == 'Scan':
        if config['parameters']['helper_sent']:
            dataset = ScanDatasetForLMs()
        else:
            dataset = ScanDataset()
    else:
        raise Exception("No valid dataset was given.")
    dataloader = DataLoader(dataset, batch_size=config['parameters']['batch_size'],
                            shuffle=False, num_workers=1)
    results = []
    model.eval()
    decoded_outputs = []
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(dataloader)):
            #if i_batch>10:
            #    break
            sample1 = batch['s1']
            label = batch['label']
            #alternatives = list(zip(*batch['alternatives']))
            if config['parameters']['helper_sent']:
                sample2 = batch['s2']
                tokenized = tokenize(sample1, sample2).to(device)
            else:
                tokenized = tokenize(sample1).to(device)

            if 'Bert' in config["parameters"]["model"]:
                mask_index_batch, mask_index = torch.where(torch.tensor(tokenized['input_ids']) == tokenizer.mask_token_id)
            elif "GPT" in config["parameters"]["model"]:
                mask_index = torch.sum(torch.tensor(tokenized['input_ids']) != tokenizer.eos_token_id, dim=1)-1
                mask_index_batch = torch.arange(0, len(tokenized['input_ids']), dtype=torch.int64).to(device)
            with torch.no_grad():
            #label = set(sample[3].split('/'))
                # model = model
                # tokenized = tokenized
                output = model(**tokenized)
                output = output.logits
                output = functional.softmax(output, dim=-1)
            mask_word = output[mask_index_batch, mask_index, :]
            _, top_k = torch.topk(mask_word, 10, dim=-1)
            #top_10 = torch.topk(mask_word, 10, dim=-1)[1][0].to('cpu')
            # label = tokenizer.decode(label['input_ids'].flatten().tolist(), skip_special_tokens=True)
            # text = tokenizer.decode(sample['input_ids'].flatten().tolist(), skip_special_tokens=True)

            #decoded_outputs = tokenizer.batch_decode(top_k)
            for k, entry in enumerate(top_k):
                words = tokenizer.decode(entry)
                found = False
                decoded_outputs.append(words.split(' '))
    with open('saved_tests/eval_{}_{}_{}_2022.pkl'.format(config['parameters']['model'], config['parameters']['dataset'],
                                                          config['parameters']['helper_sent']), 'wb') as f:
        pkl.dump(decoded_outputs, f)
    acc, mrr, recall10 = evaluate_outputs(decoded_outputs, dataset)
    return decoded_outputs


def glove_tokenize_turney(batch, model):
    words = list(zip(*batch['s1']))
    # tokenized_batch = torch.ones((len(batch['s1']), len(words)), dtype=torch.int64)
    tokenized_batch = []
    for b_id in range(len(words)):
        per_batch = []
        for w_id, w in enumerate(words[b_id]):
            try:
                w.split(' ')
            except Exception as inst:
                print(inst)
            if len(w.split(' ')) > 1:
                tmp = []
                for word in w.split(' '):
                    tmp.append(model.word2idx[word])
                per_batch.append(tmp)
            else:
                per_batch.append([model.word2idx[w]])
        tokenized_batch.append(per_batch)
    #labels = torch.tensor([model.word2idx[batch['label'][i]] for i in range(len(batch['label']))])
    return tokenized_batch


def evaluate_outputs(outputs, dataset):
    df = dataset.df
    #df['helper_row'] = outputs.keys()
    df['outputs'] = outputs
    labels = list(df['src_word']) if (isinstance(dataset, ScanDataset) or (isinstance(dataset, ScanDatasetForLMs))) else list(df['word4'])
    alternatives = list(df['alternatives'])
    if type(alternatives[0]).__name__ != 'list':
        alternatives = [str(item).split(", ") for item in alternatives]

    mrr = mean_reciprocal_rank(outputs, labels, alternatives)
    recall10 = recall_at_k(outputs, labels, alternatives, 10)
    recall5 = recall_at_k(outputs, labels, alternatives, 5)
    accuracy = calc_accuracy(outputs, labels, alternatives)
    accuracy10 = calc_accuracy(outputs, labels, alternatives, k=10)
    print("Acc@10: {}, General: Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(np.format_float_positional(accuracy10, precision=3), accuracy, mrr, recall10, recall5))

    science_outputs = list(df.loc[df['analogy_type'] == 'science']['outputs'])
    science_labels = list(df.loc[df['analogy_type'] == 'science']['src_word'])
    science_alternatives = [str(item).split(', ') for item in list(df.loc[df['analogy_type'] == 'science']['alternatives'])]
    science_mrr = mean_reciprocal_rank(science_outputs, science_labels, science_alternatives)
    science_recall10 = recall_at_k(science_outputs, science_labels, science_alternatives)
    science_recall5 = recall_at_k(science_outputs, science_labels, science_alternatives, k=5)
    science_accuracy10 = calc_accuracy(science_outputs, science_labels, science_alternatives, k=10)
    science_accuracy = calc_accuracy(science_outputs, science_labels, science_alternatives)

    metaphor_outputs = list(df.loc[df['analogy_type'] == 'metaphor']['outputs'])
    metaphor_labels = list(df.loc[df['analogy_type'] == 'metaphor']['src_word'])
    metaphor_alternatives = [str(item).split(', ') for item in list(df.loc[df['analogy_type'] == 'metaphor']['alternatives'])]
    metaphor_mrr = mean_reciprocal_rank(metaphor_outputs, metaphor_labels, metaphor_alternatives)
    metaphor_recall10 = recall_at_k(metaphor_outputs, metaphor_labels, metaphor_alternatives)
    metaphor_recall5 = recall_at_k(metaphor_outputs, metaphor_labels, metaphor_alternatives, k=5)
    metaphor_accuracy = calc_accuracy(metaphor_outputs, metaphor_labels, metaphor_alternatives)
    metaphor_accuracy10 = calc_accuracy(metaphor_outputs, metaphor_labels, metaphor_alternatives, k=10)

    print("Science: Accuracy10: {}, Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(science_accuracy10,
                                                                                             science_accuracy,
                                                                                             science_mrr,
                                                                                             science_recall10,
                                                                                             science_recall5))
    print("Metaphor: Accuracy10: {}, Accuracy: {}, MRR: {}, Recall10: {}, Recall5: {}".format(metaphor_accuracy10,
                                                                                              metaphor_accuracy,
                                                                                              metaphor_mrr,
                                                                                              metaphor_recall10,
                                                                                              metaphor_recall5))
    print("\n")
    print("\n")
    print("&{} &{} &{} &{} &{} &{} &{} &{} &{} &{} &{} &{}".format(np.format_float_positional(accuracy, precision=3),
                                                                   np.format_float_positional(mrr, precision=3),
                                                                   np.format_float_positional(recall10, precision=3),
                                                                   np.format_float_positional(recall5, precision=3),
                                                                   np.format_float_positional(science_accuracy,
                                                                                              precision=3),
                                                                   np.format_float_positional(science_mrr, precision=3),
                                                                   np.format_float_positional(science_recall10,
                                                                                              precision=3),
                                                                   np.format_float_positional(science_recall5,
                                                                                              precision=3),
                                                                   np.format_float_positional(metaphor_accuracy,
                                                                                              precision=3),
                                                                   np.format_float_positional(metaphor_mrr,
                                                                                              precision=3),
                                                                   np.format_float_positional(metaphor_recall10,
                                                                                              precision=3),
                                                                   np.format_float_positional(metaphor_recall5,
                                                                                              precision=3)
                                                                   ))
    return accuracy, mrr, recall10

def mean_reciprocal_rank(outputs, labels, alternatives):
    rec_ranks = []
    for i, sample in enumerate(outputs):
        relevant = alternatives[i]
        relevant.append(labels[i])
        relevant = (set(relevant) - {'nan'}) - {"None"}
        found = False
        for j, word in enumerate(sample):
            if word in relevant:
                rec_ranks.append(1 / (j + 1))
                found = True
                break
        if not found:
            rec_ranks.append(0)
    mrr = np.mean(rec_ranks)
    return mrr, rec_ranks


def calc_accuracy(outputs, labels, alternatives, k=1):
    predictions = []
    for i, sample in enumerate(outputs):
        relevant = alternatives[i]
        relevant.append(labels[i])
        relevant = (set(relevant) - {'nan'}) - {"None"}
        found = False
        for j, word in enumerate(sample):
            if j >= k:
                break
            #print(word, labels[i], alternatives[i])
            if word in relevant:
                found = True
                predictions.append(1)
                break
        if not found:
            predictions.append(0)
    acc = np.mean(predictions)
    return acc, predictions

def recall_at_k(outputs, labels, alternatives, k=10):
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
        found = 0
        relevant = alternatives[i]
        relevant.append(labels[i])
        relevant = (set(relevant) - {'nan'}) - {"None"}
        for word in relevant:
            if word in outputs[i][:k]:
                found += 1
        recalls.append(found/len(relevant))
    avg_recall_at_k = np.mean(recalls)
    return avg_recall_at_k, recalls


def solve_analogy_glove(config_path):
    with open("configs/{}".format(config_path)) as f:
        config = json.load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config["parameters"]["model"] == "GloveNoTraining":
        model = GloveNoTraining(config).to(device)
        tokenize = glove_tokenize_turney
    else:
        raise Exception("Could not identify model.")
    if config['parameters']['dataset'] == 'BATS':
        # create_bats_split(None)
        #train_dataset = BatsDataset('train')
        #dev_dataset = BatsDataset('dev')
        #dataset = BatsDataset('test')
        dataset = BatsDataset('all')
        print(len(dataset))
        # print(len(train_dataset), len(dev_dataset), len(dataset))
    elif config['parameters']['dataset'] == 'Turney':
        dataset = TurneyDataset()
    elif config['parameters']['dataset'] == 'Scan':
        dataset = ScanDataset()
    else:
        raise Exception("No valid dataset was given.")

    fname = 'saved_tests/eval_{}_{}_untrained_2022.pkl'.format(config['parameters']['model'], config['parameters']['dataset'])
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            outputs = pkl.load(f)
            # outputs = outputs.to(device)
    else:
        dataloader = DataLoader(dataset, batch_size=config['parameters']['batch_size'],
                                shuffle=False, num_workers=0)
        results = []
        model.eval()
        outputs = []
        count_samples = 0
        labels = []
        alternatives = {}
        print(device)
        for i_batch, batch in tqdm(enumerate(dataloader)):
            count_samples += len(batch['s1'][0])
            sample1 = np.array(batch['s1'])
            label = batch['label']
            #labels.extend(label)
            #alternatives = list(zip(*batch['alternatives']))
            alternative = batch['alternatives']
            tokenized = tokenize(batch, model)
            with torch.no_grad():
                # label = set(sample[3].split('/'))
                output = model(tokenized)
            k = 10
            xq = output.numpy().astype(dtype='float32')
            D, I = model.index.search(xq, k)
            for i, row in enumerate(I):
                found = False
                words = [model.idx2word[idx] for idx in row]
                alternatives[(sample1[0, i], sample1[1, i], sample1[2, i])] = alternative
                #if (sample1[0, i], sample1[1, i], sample1[2, i]) in outputs:
                #    #print(labels[(sample1[0, i], sample1[1, i], sample1[2, i])])
                #    alternatives[(sample1[0, i], sample1[1, i], sample1[2, i])].append(label[i])
                #    print('analogy in there')
                labels.append(label[i])
                outputs.append(words)
                #print("####################")
                #for idx in row:
                #    print(sample1[0, i], sample1[1, i], sample1[2, i], model.idx2word[idx], label[i])
                #    if model.idx2word[idx] in label[i].split(' ') or model.idx2word[idx] in alternatives[i]:
                #        found = True
                #results.append(found)
        with open('saved_tests/eval_{}_{}_untrained_2022.pkl'.format(config['parameters']['model'], config['parameters']['dataset']), 'wb') as f:
            pkl.dump(outputs, f)
    acc, mrr, recall = evaluate_outputs(list(outputs), dataset)
    print(len(results))
    acc = results.count(1) / len(results)
    print('acc: {}'.format(acc))
    calc_accs_per_type(results, dataset)
    #print('count samples: {}, batch_id: {}'.format(count_samples, i_batch))

    return acc


if __name__ == '__main__':
    accs = []
    num_iterations = 1
    for i in range(num_iterations):
        accs.append(solve_analogy_glove('config_2022.json'))
    print('final acc: {}'.format(sum(accs)/num_iterations))
