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

Accompanying code for the ARR paper "_Making Event coreference resolution Tough Again. Metaphorically speaking_"

## Contents
  1. [Getting Started](#getting-started)
  2. [Preprocessing](#preprocessing)
  3. [ECB+META Generation](#ecbmeta-generation)
  4. [Annotations](#annotations)
  5. [Training](#training)
  6. [Prediction](#prediction)
  7. [Error Analysis](#error-analysis)

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

## ECB+META Generation
### ECB+META_1
Run the following scripts to generate the corpus file for the single-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_1/mention_map.pkl`

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

### ECB+META_m
Run the following scripts to generate the corpus file for the multi-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_m/mention_map.pkl`

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

## Training

## Prediction

## Error Analysis
The file used for Error Analysis on the dev_small split of ECB+META_1 and ECB+META_m can be found at: [data/ecb_meta_analysis.xlsx](data/ecb_meta_analysis.xlsx)
