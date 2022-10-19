import numpy as np
import pickle as pkl
import scipy
from scipy.stats import permutation_test

def calc_permutation_sign():
    print(scipy.__version__)
    with open('saved_tests/sign_test_results1.pkl', "rb") as f:
        results = pkl.load(f)
    with open('saved_tests/sign_test_results2.pkl', "rb") as f:
        results2 = pkl.load(f)
    with open('saved_tests/sign_test_scores1.pkl', "rb") as f:
        scores = pkl.load(f)
    with open('saved_tests/sign_test_scores2.pkl', "rb") as f:
        scores2 = pkl.load(f)
    dataset_names = ['ScanDataset', 'ScanDatasetForLMs']
    models = ['GPT2LMHeadModel', 'BertForMaskedLM', 'MultilingualBert']
    subsets = ["normal"]
    dataset_name = dataset_names[0]
    dataset_name2 = dataset_names[1]
    if "ScanDataset" in dataset_name:
        subsets.extend(["science", "metaphor"])
    for subset in subsets:
        print(subset)
        for eval_func in results[subset].keys():
            for i in range(len(models)):
                #t_res = scipy.stats.ttest_ind(results[eval_func][models[i]], results2[eval_func][models[i]])
                t_res = permutation_test((results[subset][eval_func][models[i]], results2[subset][eval_func][models[i]]), statistic, vectorized=True,
                                                     n_resamples=10e5)
                if t_res.pvalue <= 0.05:
                    print("SIGNIFICANT T-test for {}, {}, {} and {}. Scores: {}, {} Res: {}".format(dataset_name,
                                                                                                    eval_func,
                                                                                                    models[i],
                                                                                                    models[i],
                                                                                                    np.format_float_positional(scores[subset][eval_func][models[i]], precision=3),
                                                                                                    np.format_float_positional(scores2[subset][eval_func][models[i]], precision=3),
                                                                                                    t_res))
                else:
                    print("NOT SIGNIFICANT T-test for {}, {}, {} and {}. Scores: {}, {} Res: {}".format(dataset_name,
                                                                                                        eval_func,
                                                                                                        models[i],
                                                                                                        models[i],
                                                                                                        np.format_float_positional(scores[subset][eval_func][models[i]], precision=3),
                                                                                                        np.format_float_positional(scores2[subset][eval_func][models[i]], precision=3),
                                                                                                        t_res))

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

if __name__ == "__main__":
    calc_permutation_sign()
