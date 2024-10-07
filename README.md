# Factorizer

Factor-based tokenizer. Wraps the [OpenNMT tokenizer](https://github.com/microsoft/factored-segmenter).

### Installation


```
# pip install it from github repository
pip install git+https://github.com/lingea-research/factorizer
# or clone it and install locally
git clone https://github.com/lingea-research/factorizer
cd factorizer 
pip install -e .
```

### Usage

#### Data processing

```
./factorizer.py (--tokenize | --detokenize) \
  [--src SRC_FILE_PATH] [--tgt TGT_FILE_PATH] [--constr CONSTR_FILE_PATH] \
  [--model SPM_MODEL_PATH] [--no_case_feature] [--no_preserve_placeholders] \
  [--reserved_symbols [SYMBOLS ...]] [--add_factors_soft [FACTORS ...]] \
  [--add_factors_hard [FACTORS ...]] [--silent]
```

#### SentencePiece model training

```
./factorizer.py --spm_train \
  [--train_sets [TRAIN_SETS ...]] [--vocab_size VOCAB_SIZE] \
  [--character_coverage CHARACTER_COVERAGE] [--train_extremely_large_corpus] \
  [--silent]
```
