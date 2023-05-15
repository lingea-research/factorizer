# Pyonmttok adapter

Adapter for OpenNMT pyonmttok tokenizer.

### Dependencies

```
pip install -r requirements.txt
```

### Usage

```
# provide I/O via arguments
./pyonmttok_wrapper.py -m <path to .spm vocab> -s <path to the file to (de-)tokenize> -t <path to (de-)tokenized file> (--tokenize | --detokenize)
# provide I/O via streams
cat <path to the file to (de-)tokenize> | ./pyonmttok_wrapper.py -m <path to .spm vocab> (--tokenize | --detokenize) > <path to (de-)tokenized file>
```



