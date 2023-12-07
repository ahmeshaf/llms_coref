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

### faiss installation
```shell

```


## Getting Started

```sh
python -m spacy project assets
```

```sh
python -m spacy project run ecb-setup
```

## BERT Pipelines
```shell
python -m scripts.bert_pipeline ./corpus/ecb/ debug_split ./outputs/lh/mention_pairs.pkl 
/home/rehan/workspace/models/ce_models/ecb_small/ --max-sentence-len 512 --ce-score-file 
./outputs/knn/ce_score_lh.pkl
```