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

## Prereqs

### Libraries
```shell
pip install -r requirements.txt
```

Spacy model:
```shell
python -m spacy download en_core_web_lg
```

### Change Directory to project
```shell
cd project
```

### OpenAI API Key Setup
The OpenAI API Key can be set up by the below line:
```shell
export OPENAI_API_KEY=<Your-OpenAI-API-Key>
```



## Getting Started with ECB+ Corpus

These scripts download and process the ECB+ corpus into a pkl corpus file which we call `mention_map.pkl`
```sh
python -m spacy project assets
```

```sh
python -m spacy project run ecb-setup
```

This will create the corpus file at `corpus/ecb/mention_map.pkl`

## Generate ECB+META Corpora
### ECB+META_1
Run the following scripts to generate the corpus file for the single-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_1/mention_map.pkl`

Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_single
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_single
```
Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_single/merged.pkl ./outputs/meta_single/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_single/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_single/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_single/merged.pkl  ./corpus/ecb_meta_single/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_single/mention_map.pkl
```

### ECB+META_m
Run the following scripts to generate the corpus file for the multi-word metaphoric transformation of ECB+ at: 
`corpus/ecb_meta_m/mention_map.pkl`

Run GPT-4 pipeline:
```shell
python -m scripts.llm_pipeline corpus/ecb/ test  --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ dev --experiment-name meta_multi
python -m scripts.llm_pipeline corpus/ecb/ debug_split --experiment-name meta_multi
```
Generate corpus file:
```shell
python scripts/merge_meta.py ./outputs/meta_multi/merged.pkl ./outputs/meta_multi/gpt-4*.pkl
python -m scripts.parse_meta save-doc-sent-map ./outputs/meta_multi/merged.pkl ./corpus/ecb/doc_sent_map.pkl ./corpus/ecb_meta_multi/doc_sent_map.pkl
python -m scripts.parse_meta parse ./outputs/meta_multi/merged.pkl  ./corpus/ecb_meta_multi/doc_sent_map.pkl ./corpus/ecb/mention_map.pkl ./corpus/ecb_meta_multi/mention_map.pkl

```

## BERT Pipelines
```shell
python -m scripts.bert_pipeline ./corpus/ecb/ debug_split ./outputs/lh/mention_pairs.pkl 
/home/rehan/workspace/models/ce_models/ecb_small/ --max-sentence-len 512 --ce-score-file 
./outputs/knn/ce_score_lh.pkl
```