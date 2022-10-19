import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from test_model import calc_accuracy, mean_reciprocal_rank, recall_at_k, tokenize_mult_bert_helper, mult_bert_tokenize, tokenize_bert_helper, bert_tokenize, tokenize_gpt2_helper, gpt2_tokenize, prepare_alternatives_bert, prepare_alternatives_gpt, CPU_Unpickler
from transformers import BertTokenizer, GPT2Tokenizer
from data.datasets import BatsDataset, BatsDatasetForLMs, ScanDataset, ScanDatasetForLMs
import torch
from scipy.stats import ttest_ind
from predict_word_analogy import calc_accuracy as calc_accuracy_glove
from predict_word_analogy import mean_reciprocal_rank as mean_reciprocal_rank_glove
from predict_word_analogy import recall_at_k as recall_at_k_glove




def compare_results(file_names, dataset):
    dicts = []
    for file_name in file_names:
        with open('saved_tests/{}.pkl'.format(file_name), 'rb') as f:
            dicts.append(pkl.load(f))
    df = dataset.df
    print(len(df), len(dicts[0]), len(dicts[1]), len(dicts[2]))
    for i, file_name in enumerate(file_names):
        df[file_name] = dicts[i]
    all_same = df[df[[file_names[0], file_names[1], file_names[2]]].nunique(axis=1) == 1]
    all_pos = all_same.loc[all_same[file_names[0]] == 1]
    all_neg = all_same.loc[all_same[file_names[0]] == 0]
    gpt2 = df.loc[(df[file_names[-1]]==1)] #& (df[file_names[-2]]==0)]
    print('here')

def save_results():
    device = 'cpu'
    helper_sent = 0
    dataset_names = ['ScanDataset', 'ScanDataset']
    models = ['GPT2LMHeadModel', 'BertForMaskedLM', 'MultilingualBert']
    eval_funcs = [calc_accuracy, mean_reciprocal_rank, recall_at_k, recall_at_k]
    subsets = ["normal"]
    dataset_name = dataset_names[0]
    results = {}
    results2 = {}
    scores = {}
    scores2 = {}
    print('DATASET: {}'.format(dataset_name))
    if "ScanDataset" in dataset_name:
        subsets.extend(["science", "metaphor"])
    for subset in subsets:
        results[subset] = {}
        print("SUBSET: {}".format(subset))
        #results = {}
        results2[subset] = {}
        scores[subset] = {}
        scores2[subset] = {}
        first_recall_at_k = True
        for eval_func in eval_funcs:
            if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                results[subset]["recall_at_10"] = {}
                results2[subset]["recall_at_10"] = {}
                scores[subset]["recall_at_10"] = {}
                scores2[subset]["recall_at_10"] = {}
            elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                results[subset]["recall_at_5"] = {}
                results2[subset]["recall_at_5"] = {}
                scores[subset]["recall_at_5"] = {}
                scores2[subset]["recall_at_5"] = {}
            else:
                results[subset][eval_func.__name__]= {}
                results2[subset][eval_func.__name__] = {}
                scores[subset][eval_func.__name__] = {}
                scores2[subset][eval_func.__name__] = {}
            for model in models:
                helper_sent = 0
                #file1 = 'saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model, dataset_name, 0)
                file1 = 'saved_tests/eval_GloveNoTraining_Scan_untrained_2022.pkl'
                print(file1)
                with open(file1, 'rb') as f:
                    if device == "cpu":
                        saved_outputs = CPU_Unpickler(f).load()
                    else:
                        saved_outputs = pkl.load(f)
                tokenize, tokenizer = choose_tokenizer(model, helper_sent)
                outputs, labels, alternatives = get_data(saved_outputs, dataset_name, helper_sent, tokenizer, model, subset, device)
                if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results[subset]["recall_at_10"][model] = out
                    scores[subset]["recall_at_10"][model] = score
                elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer, k=5)
                    results[subset]["recall_at_5"][model] = out
                    scores[subset]["recall_at_5"][model] = score
                else:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results[subset][eval_func.__name__][model] = out
                    scores[subset][eval_func.__name__][model] = score
            dataset_name2 = dataset_names[0]
            for model in models:
                helper_sent = 1
                file2 = 'saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model, dataset_name2, helper_sent)
                print(file2)
                with open(file2, 'rb') as f:
                    if device == "cpu":
                        saved_outputs = CPU_Unpickler(f).load()
                    else:
                        saved_outputs = pkl.load(f)
                tokenize, tokenizer = choose_tokenizer(model, helper_sent)
                outputs, labels, alternatives = get_data(saved_outputs, dataset_name2, helper_sent, tokenizer, model,
                                                         subset, device)
                if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results2[subset]["recall_at_10"][model] = out
                    scores2[subset]["recall_at_10"][model] = score
                elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer, k=5)
                    results2[subset]["recall_at_5"][model] = out
                    scores2[subset]["recall_at_5"][model] = score
                else:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results2[subset][eval_func.__name__][model] = out
                    scores2[subset][eval_func.__name__][model] = score
            if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                first_recall_at_k = False

    with open('saved_tests/sign_test_results1.pkl', "wb") as f:
        pkl.dump(results, f)
    with open('saved_tests/sign_test_results2.pkl', "wb") as f:
        pkl.dump(results2, f)
    with open('saved_tests/sign_test_scores1.pkl', "wb") as f:
        pkl.dump(scores, f)
    with open('saved_tests/sign_test_scores2.pkl', "wb") as f:
        pkl.dump(scores2, f)


def get_data(outputs, dataset_name, helper_sent, tokenizer, model_name, subset="normal", device='cpu'):
    dataset = choose_dataset(dataset_name, helper_sent)
    df = dataset.df
    df['outputs'] = outputs
    labels = list(df['src_word']) if (
                isinstance(dataset, ScanDataset) or (isinstance(dataset, ScanDatasetForLMs))) else list(df['word4'])
    if 'GPT' in type(tokenizer).__name__:
        vocab = tokenizer.get_vocab()
        labels = torch.tensor(
            [vocab['Ġ' + word] if ('Ġ' + word) in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
             for word in labels]).to(device)
    elif "Bert" in type(tokenizer).__name__:
        vocab = tokenizer.get_vocab()
        labels = torch.tensor(
            [vocab[word] if word in vocab else tokenizer(word, add_special_tokens=False)['input_ids'][0]
             for word in labels]).to(device)
    if type(labels).__name__ != 'list':
        df['tok_labels'] = labels.cpu()
    else:
        df['tok_labels'] = labels
    alternatives = [str(item) for item in list(df['alternatives'])]
    if "Bert" in type(tokenizer).__name__:
        tok_alternatives = prepare_alternatives_bert(tokenizer, alternatives, device)
    elif 'GPT' in type(tokenizer).__name__:
        tok_alternatives = prepare_alternatives_gpt(tokenizer, alternatives, device)
    else:
        tok_alternatives = [item.split(', ') for item in alternatives]
    if type(tok_alternatives).__name__ != 'list':
        df['tok_alternatives'] = tok_alternatives.cpu()
    else:
        df['tok_alternatives'] = tok_alternatives
    if subset=="normal":
        return outputs, labels, tok_alternatives
    elif subset=="science":
        science_outputs = list(df.loc[df['analogy_type'] == 'science']['outputs'])
        science_labels = list(df.loc[df['analogy_type'] == 'science']['tok_labels'])
        if ("GPT" in type(tokenizer).__name__) or ("Bert" in type(tokenizer).__name__):
            science_alternatives = torch.tensor(list(df.loc[df['analogy_type'] == 'science']['tok_alternatives']))
        else:
            science_alternatives = list(df.loc[df['analogy_type'] == 'science']['tok_alternatives'])
        return science_outputs, science_labels, science_alternatives
    elif subset=="metaphor":
        metaphor_outputs = list(df.loc[df['analogy_type'] == 'metaphor']['outputs'])
        metaphor_labels = list(df.loc[df['analogy_type'] == 'metaphor']['tok_labels'])
        if ("GPT" in type(tokenizer).__name__) or ("Bert" in type(tokenizer).__name__):
            metaphor_alternatives = torch.tensor(list(df.loc[df['analogy_type'] == 'metaphor']['tok_alternatives']))
        else:
            metaphor_alternatives = list(df.loc[df['analogy_type'] == 'metaphor']['tok_alternatives'])
        return metaphor_outputs, metaphor_labels, metaphor_alternatives
    else:
        raise Exception("Invalid subset selected")



def choose_dataset(dataset_name, helper_sent):
    if dataset_name == 'BatsDataset':
        #create_strict_bats_split('all', alts=True)
        if helper_sent:
            dataset = BatsDatasetForLMs('all')
            #dataset.df = dataset.df.dropna()
        else:
            dataset = BatsDataset('all')
            #dataset.df = dataset.df.dropna()
            #dataset = BatsDataset(mode='test', strict=True)
    elif dataset_name == 'ScanDataset':
        if helper_sent:
            dataset = ScanDatasetForLMs()
        else:
            dataset = ScanDataset()
    else:
        raise Exception("No valid dataset was given.")
    return dataset


def choose_tokenizer(model, helper_sent):
    if model == "BertForMaskedLM":
        if helper_sent:
            tokenize = tokenize_bert_helper
        else:
            tokenize = bert_tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model == "MultilingualBert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if helper_sent:
            tokenize = tokenize_mult_bert_helper
        else:
            tokenize = mult_bert_tokenize
    elif model == "GPT2LMHeadModel":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        if helper_sent:
            tokenize = tokenize_gpt2_helper
        else:
            tokenize = gpt2_tokenize
    else:
        raise Exception("No valid model given!")
    return tokenize, tokenizer


def save_results_glove():
    device = 'cpu'
    helper_sent = 0
    dataset_names = ['ScanDataset', 'ScanDataset']
    models = ['GPT2LMHeadModel', 'BertForMaskedLM', 'MultilingualBert']
    eval_funcs = [calc_accuracy, mean_reciprocal_rank, recall_at_k, recall_at_k]
    subsets = ["normal"]
    glove_funcs = {calc_accuracy: calc_accuracy_glove, mean_reciprocal_rank: mean_reciprocal_rank_glove, recall_at_k: recall_at_k_glove}
    dataset_name = dataset_names[0]
    results = {}
    results2 = {}
    scores = {}
    scores2 = {}
    print('DATASET: {}'.format(dataset_name))
    if "ScanDataset" in dataset_name:
        subsets.extend(["science", "metaphor"])
    for subset in subsets:
        results[subset] = {}
        print("SUBSET: {}".format(subset))
        #results = {}
        results2[subset] = {}
        scores[subset] = {}
        scores2[subset] = {}
        first_recall_at_k = True
        for eval_func in eval_funcs:
            if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                results[subset]["recall_at_10"] = {}
                results2[subset]["recall_at_10"] = {}
                scores[subset]["recall_at_10"] = {}
                scores2[subset]["recall_at_10"] = {}
            elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                results[subset]["recall_at_5"] = {}
                results2[subset]["recall_at_5"] = {}
                scores[subset]["recall_at_5"] = {}
                scores2[subset]["recall_at_5"] = {}
            else:
                results[subset][eval_func.__name__]= {}
                results2[subset][eval_func.__name__] = {}
                scores[subset][eval_func.__name__] = {}
                scores2[subset][eval_func.__name__] = {}
            for model in models:
                helper_sent = 0
                #file1 = 'saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model, dataset_name, 0)
                file1 = 'saved_tests/eval_GloveNoTraining_Scan_untrained_2022.pkl'
                print(file1)
                with open(file1, 'rb') as f:
                    if device == "cpu":
                        saved_outputs = CPU_Unpickler(f).load()
                    else:
                        saved_outputs = pkl.load(f)
                tokenize, tokenizer = choose_tokenizer(model, helper_sent)
                tokenizer = None
                outputs, labels, alternatives = get_data(saved_outputs, dataset_name, helper_sent, tokenizer, model, subset, device)
                if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                    score, out = glove_funcs[eval_func](outputs, labels, alternatives)
                    results[subset]["recall_at_10"][model] = out
                    scores[subset]["recall_at_10"][model] = score
                elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                    score, out = glove_funcs[eval_func](outputs, labels, alternatives, k=5)
                    results[subset]["recall_at_5"][model] = out
                    scores[subset]["recall_at_5"][model] = score
                else:
                    score, out = glove_funcs[eval_func](outputs, labels, alternatives)
                    results[subset][eval_func.__name__][model] = out
                    scores[subset][eval_func.__name__][model] = score
            dataset_name2 = dataset_names[0]
            for model in models:
                helper_sent = 0
                file2 = 'saved_tests/eval_{}_{}_untrained_{}_2022.pkl'.format(model, dataset_name2, helper_sent)
                print(file2)
                with open(file2, 'rb') as f:
                    if device == "cpu":
                        saved_outputs = CPU_Unpickler(f).load()
                    else:
                        saved_outputs = pkl.load(f)
                tokenize, tokenizer = choose_tokenizer(model, helper_sent)
                outputs, labels, alternatives = get_data(saved_outputs, dataset_name2, helper_sent, tokenizer, model,
                                                         subset, device)
                if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results2[subset]["recall_at_10"][model] = out
                    scores2[subset]["recall_at_10"][model] = score
                elif eval_func.__name__ == "recall_at_k" and not first_recall_at_k:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer, k=5)
                    results2[subset]["recall_at_5"][model] = out
                    scores2[subset]["recall_at_5"][model] = score
                else:
                    score, out = eval_func(outputs, labels, alternatives, tokenizer)
                    results2[subset][eval_func.__name__][model] = out
                    scores2[subset][eval_func.__name__][model] = score
            if eval_func.__name__ == "recall_at_k" and first_recall_at_k:
                first_recall_at_k = False

    with open('saved_tests/sign_test_results1.pkl', "wb") as f:
        pkl.dump(results, f)
    with open('saved_tests/sign_test_results2.pkl', "wb") as f:
        pkl.dump(results2, f)
    with open('saved_tests/sign_test_scores1.pkl', "wb") as f:
        pkl.dump(scores, f)
    with open('saved_tests/sign_test_scores2.pkl', "wb") as f:
        pkl.dump(scores2, f)


if __name__ == '__main__':
    #bats_files = ['eval_GloveNoTraining_BATS', 'eval_MultilingualBert_BatsDataset_base_0', 'eval_BertForMaskedLM_BatsDataset_base_0']#,
    #         'eval_GPT2LMHeadModel_BatsDataset_base_0']
    #scan_files = ['eval_GloveNoTraining_Scan', 'eval_MultilingualBert_ScanDataset_base_0',
    #              'eval_BertForMaskedLM_ScanDataset_base_0', 'eval_GPT2LMHeadModel_ScanDataset_base_0']
    #dataset = ScanDataset()
    #compare_results(scan_files, dataset)
    #plot_results()
    save_results()
    save_results_glove()
