#!/usr/bin/env python3
import argparse
import sentencepiece as spm
import regex as re
import sys

from factorizer.patterns import (
    special_symbol_or_byte_pattern,
    byte_seq_pattern,
    continuous_scripts_pattern,
)

header = [
    "# factors",
    "",
    "_lemma",
    "",
    "_c",
    "ci : _c",
    "ca : _c",
    "cn : _c",
    "_has_c",
    "",
    "_cb",
    "cb : _cb",
    "cbn : _cb",
    "_has_cb",
    "",
    "_class",
    "classphrasefix : _class",
    "_has_class",
    "",
    "_gl",
    "gl+ : _gl",
    "gl- : _gl",
    "_has_gl",
    "",
    "_gr",
    "gr+ : _gr",
    "gr- : _gr",
    "_has_gr",
    "",
    "_sc",
    "scu : _sc",
    "scl : _sc",
    "_has_sc",
    "",
    "_wb",
    "wb : _wb",
    "wbn : _wb",
    "_has_wb",
]


static_lemmas = [
    "{continuousScript} : _lemma _has_cb _has_class",
    "{punctuation} : _lemma _has_class _has_gl _has_gr",
    "{unk,cb} : _lemma _has_cb",
    "{unk,gl,gr} : _lemma _has_gl _has_gr",
    "{unk,sc,wb} : _lemma _has_sc _has_wb",
    "{unk,wb} : _lemma _has_wb",
    "{word-wo-case} : _lemma _has_class _has_wb",
    "{word} : _lemma _has_c _has_class _has_wb",
    "<#> : _lemma",
    "<0> : _lemma",
    "<1> : _lemma",
    "<2> : _lemma",
    "<3> : _lemma",
    "<4> : _lemma",
    "<5> : _lemma",
    "<6> : _lemma",
    "<7> : _lemma",
    "<8> : _lemma",
    "<9> : _lemma",
]


footer = [
    "",
    "# factors distributions",
    "",
    "_lemma <->",
    "_c <-> _has_c",
    "_wb <-> _has_wb",
    "_class <-> _has_class",
    "_cb <-> _has_cb",
    "_gl <-> _has_gl",
    "_gr <-> _has_gr",
    "_sc <-> _has_sc",
]


reserved_symbols = ["#", ":", "_", "\\", "|", "▁"]


def add_factors(lemma, add_constraints):
    factored_lemma = (
        f"{lemma} : _lemma"
        if re.search(special_symbol_or_byte_pattern, lemma)
        else f"{lemma} : _lemma _has_gl _has_gr"
        if re.search(byte_seq_pattern, lemma)
        else f"{lemma} : _lemma _has_cb"
        if re.search(continuous_scripts_pattern, lemma)
        else f"{lemma.upper()} : _lemma _has_wb"
        if lemma.isalpha() and lemma.upper() == lemma.lower()
        else f"{lemma} : _lemma _has_wb"
        if lemma.isnumeric()
        else f'{lemma.upper()} : _lemma _has_{"s" if len(lemma) == 1 else ""}c _has_wb'
        if lemma.isalpha()
        else f"{lemma} : _lemma _has_gl _has_gr"
    )
    if add_constraints and not re.search(r"^\<.*\>$", lemma):
        factored_lemma += " _has_t"
    return factored_lemma


def format_vocab(vocab, add_constraints):
    new_vocab = set()
    # add factors that do not contain special symbols
    new_vocab.update(
        [
            add_factors(lemma, add_constraints)
            for lemma in vocab
            if all(
                [reserved_symbol not in lemma for reserved_symbol in reserved_symbols]
            )
        ]
    )
    new_vocab.update(static_lemmas)
    new_vocab = sorted(new_vocab)
    return new_vocab


def convert(source, target, add_constraints):
    global header, static_lemmas, footer
    sp = spm.SentencePieceProcessor()
    sp.load(source)

    if add_constraints:
        header += [
            "",
            "_t",
            "t0 : _t",
            "t1 : _t",
            "t2 : _t",
            "_has_t",
        ]
        for idx, static_lemma in enumerate(static_lemmas[:8]):
            static_lemmas[idx] = f"{static_lemma} _has_t"
        footer += ["_t <-> _has_t"]
    header += ["", "# lemmas", ""]

    vocab = format_vocab(
        [
            sp.IdToPiece(id).strip("▁")
            for id in range(sp.GetPieceSize())
            if sp.IdToPiece(id).strip("▁")
        ],
        add_constraints,
    )

    with open(
        target, "w"
    ) if args.target is not sys.stdout else sys.stdout as target_file:
        for line in [*header, *vocab, *footer]:
            target_file.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tool for .spm to .fsv vocab conversion")
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        type=str,
        required=True,
        help="path to .spm vocab to be converted",
    )
    parser.add_argument(
        "-t",
        "--target",
        default=sys.stdout,
    )
    parser.add_argument(
        "--add_constraints",
        action="store_true",
        default=False,
        help="Adds constraint factors (|t0, |t1, |t2) to the vocabulary",
    )
    args = parser.parse_args()
    convert(**args.__dict__)
