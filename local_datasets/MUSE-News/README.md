---
license: cc-by-4.0
dataset_info:
- config_name: knowmem
  features:
  - name: answer
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: retain_qa_icl
    num_bytes: 1265
    num_examples: 10
  - name: retain_qa
    num_bytes: 11425
    num_examples: 100
  - name: forget_qa
    num_bytes: 11019
    num_examples: 100
  - name: forget_qa_icl
    num_bytes: 1154
    num_examples: 10
  download_size: 26647
  dataset_size: 24863
- config_name: privleak
  features:
  - name: text
    dtype: string
  splits:
  - name: retain
    num_bytes: 808029
    num_examples: 100
  - name: forget
    num_bytes: 806172
    num_examples: 100
  - name: holdout
    num_bytes: 806341
    num_examples: 100
  download_size: 1485975
  dataset_size: 2420542
- config_name: raw
  features:
  - name: text
    dtype: string
  splits:
  - name: retain2
    num_bytes: 6437312
    num_examples: 1778
  - name: forget
    num_bytes: 3281546
    num_examples: 889
  - name: retain1
    num_bytes: 6456895
    num_examples: 1777
  - name: holdout
    num_bytes: 9095347
    num_examples: 3043
  download_size: 14881285
  dataset_size: 25271100
- config_name: scal
  features:
  - name: text
    dtype: string
  splits:
  - name: forget_3
    num_bytes: 9874079
    num_examples: 2667
  - name: forget_2
    num_bytes: 6551494
    num_examples: 1778
  - name: forget_4
    num_bytes: 13219611
    num_examples: 3554
  - name: forget_1
    num_bytes: 3281546
    num_examples: 889
  download_size: 19240874
  dataset_size: 32926730
- config_name: sust
  features:
  - name: text
    dtype: string
  splits:
  - name: forget_3
    num_bytes: 3322585
    num_examples: 889
  - name: forget_2
    num_bytes: 3269948
    num_examples: 889
  - name: forget_4
    num_bytes: 3345532
    num_examples: 887
  - name: forget_1
    num_bytes: 3281546
    num_examples: 889
  download_size: 7721364
  dataset_size: 13219611
- config_name: train
  features:
  - name: text
    dtype: string
  splits:
  - name: retain2
    num_bytes: 6437312
    num_examples: 1778
  - name: forget
    num_bytes: 13219611
    num_examples: 3554
  - name: retain1
    num_bytes: 6456895
    num_examples: 1777
  download_size: 15207155
  dataset_size: 26113818
- config_name: verbmem
  features:
  - name: prompt
    dtype: string
  - name: gt
    dtype: string
  splits:
  - name: forget
    num_bytes: 451863
    num_examples: 100
  download_size: 295284
  dataset_size: 451863
configs:
- config_name: knowmem
  data_files:
  - split: retain_qa_icl
    path: knowmem/retain_qa_icl-*
  - split: retain_qa
    path: knowmem/retain_qa-*
  - split: forget_qa
    path: knowmem/forget_qa-*
  - split: forget_qa_icl
    path: knowmem/forget_qa_icl-*
- config_name: privleak
  data_files:
  - split: retain
    path: privleak/retain-*
  - split: forget
    path: privleak/forget-*
  - split: holdout
    path: privleak/holdout-*
- config_name: raw
  data_files:
  - split: retain2
    path: raw/retain2-*
  - split: forget
    path: raw/forget-*
  - split: retain1
    path: raw/retain1-*
  - split: holdout
    path: raw/holdout-*
- config_name: scal
  data_files:
  - split: forget_3
    path: scal/forget_3-*
  - split: forget_2
    path: scal/forget_2-*
  - split: forget_4
    path: scal/forget_4-*
  - split: forget_1
    path: scal/forget_1-*
- config_name: sust
  data_files:
  - split: forget_3
    path: sust/forget_3-*
  - split: forget_2
    path: sust/forget_2-*
  - split: forget_4
    path: sust/forget_4-*
  - split: forget_1
    path: sust/forget_1-*
- config_name: train
  data_files:
  - split: retain2
    path: train/retain2-*
  - split: forget
    path: train/forget-*
  - split: retain1
    path: train/retain1-*
- config_name: verbmem
  data_files:
  - split: forget
    path: verbmem/forget-*
---

# MUSE-News

MUSE is a comprehensive machine unlearning evaluation benchmark that assesses six key properties for unlearned models: (1) no verbatim memorization, (2) no knowledge memorization, (3) no privacy leakage, (4) utility preservation on data not intended for removal, (5) scalability with respect to the size of removal requests, and (6) sustainability over sequential unlearning requests. MUSE focuses on two types of textual data that commonly require unlearning: news articles (News) and novels (Books). __This repository contains the News corpus of MUSE (MUSE-News), which comprises BBC articles collected post-August 2023__.

## Details on Subsets & Splits

MUSE-News consists of 7 subsets: `raw`, `verbmem`, `knowmem`, `privleak`, `scal`, `sust`, and `train`.
- `raw`: A raw corpus from which all subsets except `scal` and `sust` are derived. The splits are:
    - `forget`: Data intended to be forgotten
    - `retain1`: Data used optionally as a calibrator for unlearning
    - `retain2`: Retain set, i.e. data seen by the target model and used for evaluation
    - `holdout`: Data never seen by the target model during pre-training and unlearning
- `verbmem`: Evaluates __verbatim memorization (C1)__. It contains a single split `forget` with 100 samples verbatim extracted from the `forget` split of the `raw` subset, each up to 2048 tokens long according to LLaMA's tokenization.
- `knowmem`: Evaluates __knowledge memorization (C2)__ and __utility preservation (C4)__. Partitioned into 2 splits: `forget_qa` set (for evaluating forget quality) and `retain_qa` set (for evaluating model utility).  Each split contains 100 question-answer pairs testing the model's knowledge on that specific split of the `raw` subset.
- `scal`: Contains forget sets used to evaluate scalability. The splits are `forget_1`, `forget_2`, `forget_3`, and `forget_4` such that `forget_2` contains `forget_1`, `forget_3` contains `forget_2`, etc.
- `sust`: Contains forget sets used to evaluate sustainability. The splits are `forget_1`, `forget_2`, `forget_3`, and `forget_4` such that all the splits are pair-wise disjoint.
- `train`: Data used for pre-training the target model.

## Loading the datasets

To load the dataset, specify the subset and the split as follows:
```py
from datasets import load_dataset

SUBSET = "verbmem"
SPLIT = "forget"
dataset = load_dataset("muse-bench/MUSE-Books", SUBSET, split=SPLIT)
```

## Applicability

Evaluating with our dataset applies to any unlearning method performed on our [target model](https://huggingface.co/swj0419/bbc-original_STEP0000100_5-31) with respect to the forget set provided in the `raw` subset (or `scal` or `sust` for scalability and sustainability).

## Codebase

For evaluating unlearning methods on our datasets, visit our [GitHub repository](https://github.com/jaechan-repo/muse_bench).

## Citing our work
