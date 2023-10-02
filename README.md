# Factored tokenizer

Factor-based tokenizer. Wraps the [OpenNMT tokenizer](https://github.com/microsoft/factored-segmenter).

### Installation

To install this package you should:

```
# install it from the gitlab repository with pip
pip install git+https://gitlab.lingea.cz/lingea/nlptools/factored_tokenizer
# or clone it and install locally
git clone https://gitlab.lingea.cz/lingea/nlptools/factored_tokenizer
cd factored_tokenizer
pip install .
```

Either pip install [this](https://gitlab.lingea.cz/lingea/nlptools/factored_tokenizer) package or clone it and run `pip install -e .`.

### Usage

#### Data processing

```
./factored_tokenizer.py (--tokenize | --detokenize) \
  [--src SRC_FILE_PATH] [--tgt TGT_FILE_PATH] [--constr CONSTR_FILE_PATH] \
  [--model SPM_MODEL_PATH] [--add_constr] [--no_case_feature] [--add_in] \
  [--silent]
```

#### SentencePiece model training

```
./factored_tokenizer.py --spm_train \
  [--train_sets [TRAIN_SETS ...]] \
  [--vocab_size VOCAB_SIZE] [--character_coverage CHARACTER_COVERAGE] [--train_extremely_large_corpus]
```
