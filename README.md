<div align='center'>
  <a href='LICENSE'>
    <img src='https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge'>
  </a>
</div>

# Event Coreference Resolution with LLMs

Modeling code adapted from:
1. [aviclu/CDLM](https://github.com/aviclu/CDLM)
2. [ahmeshaf/lemma_ce_coref](https://github.com/ahmeshaf/lemma_ce_coref)
3. [Helw150/Entity-Mover-Distance](https://github.com/Helw150/Entity-Mover-Distance)

Accompanying code for the ACL 2024 short paper "_Making Event coreference resolution Tough Again. Metaphorically speaking_"

## Contents
  1. [Getting Started](#getting-started)
  2. [Preprocessing](#preprocessing)
  3. [ECB+META Generation](#ecbmeta-generation)
  4. [Annotations](#annotations)
  5. [BiEncoder](#biencoder)
  6. [Lemma Heuristic](#lemma-heuristic)
  7. [Cross-encoder](#cross-encoder)
  8. [Prediction](#prediction)
  9. [Error Analysis](#error-analysis)


## Getting Started
- Install the required packages:

```shell
pip install -r requirements.txt
```

- Spacy model:
```shell
python -m spacy download en_core_web_lg
```
- Change Directory to project
```shell
cd project
```

- OpenAI API Key Setup
The OpenAI API Key can be set up by the below line:
```shell
export OPENAI_API_KEY=<Your-OpenAI-API-Key>
```


## Preprocessing

- These scripts download and process the ECB+ corpus into a pkl corpus file which we call `mention_map.pkl`
```sh
python -m spacy project assets
```
- Preprocess the ECB+ corpus
```sh
python -m spacy project run ecb-setup
```

This will create the corpus file at `corpus/ecb/mention_map.pkl`

### Data Format
Each mention in the corpus file is represented as follows: 
```shell
{
  "mention_id": "12_10ecb.xml_5",
  "topic": "12",
  "doc_id": "12_10ecb.xml",
  "sentence_id": "0",
  "marked_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
  "marked_doc": "The Indian navy has <m> captured </m> 23 Somalian ...",
  "mention_text": "captured",
  "lemma": "capture",
  "men_type": "evt",
  "gold_cluster": "ACT17403639225065902",
  "sentence": "The Indian navy has captured 23 Somalian pirates .",
  "start_char": 20,
  "end_char": 28,
  "neighbors_left": [],
  "neighbors_right": [sentence_1, sentence_2, ...]
}
```

## ECB+META Generation
### ECB+META<sub>1</sub>
Run the following scripts to generate the corpus file for the single-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_single/mention_map.pkl`

- Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_single
```
- Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_single/merged.pkl ./outputs/meta_single/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_single/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_single/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_single/merged.pkl  ./corpus/ecb_meta_single/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_single/mention_map.pkl
```

### ECB+META<sub>_m_</sub>
Run the following scripts to generate the corpus file for the multi-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_multi/mention_map.pkl`

- Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_multi
```
- Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_multi/merged.pkl ./outputs/meta_multi/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_multi/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_multi/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_multi/merged.pkl  ./corpus/ecb_meta_multi/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_multi/mention_map.pkl

```

## Annotations

## BiEncoder
- Training the BiEncoder KNN
```shell
python -m scripts.bert.train train-biencoder
        ./corpus/ecb/mention_map.pkl
        --model-name roberta-base
        --text-key neighbors_3
        --learning-rate 0.00001
        --batch-size 4
        --epochs 20
        --save-path ./outputs/biencoder/triplet
```
- Extracing KNN mention pairs for `ecb|ecb_meta_single|ecb_meta_multi`
```shell
 python -m scripts.knn save-knn-mention-pairs-all 
        ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl 
        ./outputs/biencoder/triplet 
        ./outputs/ecb|ecb_meta_single|ecb_meta_multi/knn/ 
        ./outputs/ecb|ecb_meta_single|ecb_meta_multi/mention_pairs/ 
        --text-key neighbors_3 
        --top-k 5
```

## Lemma Heuristic
- Extracting LH mention pairs for `ecb|ecb_meta_single|ecb_meta_multi`
```shell
python -m scripts.heuristic
       ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl
       lh
       0.1
       ./outputs/mention_pairs/ecb|ecb_meta_single|ecb_meta_multi/lh/
```
## Cross-encoder
- Training the Cross-encoder on the BiEncoder's KNN mention pairs (CE_KNN):
```shell
python -m scripts.bert.train train-cross-encoder 
       ./corpus/ecb/ 
       ./outputs/mention_pairs/ecb/knn/train.pairs 
       ./outputs/mention_pairs/ecb/knn/dev.pairs 
       ./outputs/cross_encoder/knn/ 
       --text-key neighbors_3
```
- Training the Cross-encoder on the LH's mention pairs (CE_LH):
```shell
python -m scripts.bert.train train-cross-encoder 
       ./corpus/ecb/ 
       ./outputs/mention_pairs/ecb/lh/train.pairs 
       ./outputs/mention_pairs/ecb/lh/dev.pairs 
       ./outputs/cross_encoder/lh/ 
       --text-key marked_sentence
```

## Prediction
- Running the Oracle Results for Recall Measure for `ecb|ecb_meta_single|ecb_meta_multi` on `knn|lh`
```shell
python -m scripts.heuristics_pipeline 
       ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl 
       ./outputs/mention_pairs/ecb|ecb_meta_single|ecb_meta_multi/knn|lh/ 
       --oracle
```
- Running the LH coreference Pipeline for `ecb|ecb_meta_single|ecb_meta_multi`
```shell
python -m scripts.heuristics_pipeline 
       ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl 
       ./outputs/mention_pairs/ecb|ecb_meta_single|ecb_meta_multi/lh/ 
```
- Running the CE_LH coreference pipelines on `ecb|ecb_meta_single|ecb_meta_multi` on `dev|debug_split|test`
```shell
python -m scripts.bert_pipeline 
       ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl 
       dev|debug_split|test 
       ./outputs/mention_pairs/ecb/lh/dev|debug_split|test.pairs 
       ./outputs/crossencoder/lh 
       --text-key marked_sentence 
       --max-sentence-length 256
```
- Running the CE_KNN coreference pipelines on `ecb|ecb_meta_single|ecb_meta_multi` on `dev|debug_split|test`
```shell
python -m scripts.bert_pipeline 
       ./corpus/ecb|ecb_meta_single|ecb_meta_multi/mention_map.pkl 
       dev|debug_split|test 
       ./outputs/mention_pairs/ecb/knn/dev|debug_split|test.pairs 
       ./outputs/crossencoder/knn 
       --text-key neighbors_3 
       --max-sentence-length 512
```

## Error Analysis
The file used for Error Analysis on the dev_small split of ECB+META_1 and ECB+META_m can be found at: [data/ecb_meta_analysis.xlsx](data/ecb_meta_analysis.xlsx)
