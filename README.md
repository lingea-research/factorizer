# Factored tokenizer

Factor-based tokenizer. Wraps the OpenNMT tokenizer.
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

Either pip install [this](https://gitlab.lingea.cz/lingea/nlptools/factored_tokenizer) package or clone it and run `pip install .`.

### Usage

```
# provide I/O via arguments
./factored_tokenizer.py -m <path to .spm vocab> -s <path to the file to (de-)tokenize> -t <path to (de-)tokenized file> (--tokenize | --detokenize)
# provide I/O via streams
./factored_tokenizer.py -m <path to .spm vocab> (--tokenize | --detokenize) < <path to the file to (de-)tokenize> > <path to (de-)tokenized file>
```



