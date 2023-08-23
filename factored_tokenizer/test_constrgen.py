import sys
import argparse

from pyonmttok_wrapper import PyonmttokWrapper


def gen(src_path, tgt_path, vocab_path, model_path, out_path):
    tokenzier = PyonmttokWrapper(model=model_path)
    constrs = tokenzier.generate_constraints(
        src_path, tgt_path, vocab_path, use_lemmatization=True, constraints_file=out_path, src_lang='cs', tgt_lang='de'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", dest="model_path")
    parser.add_argument("--src", "-s", dest="src_path")
    parser.add_argument("--tgt", "-t", dest="tgt_path")
    parser.add_argument("--vocab", "-v", dest="vocab_path")
    parser.add_argument("--out", "-o", dest="out_path", default=sys.stdout)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen(**args.__dict__)
