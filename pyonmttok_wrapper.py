#!/usr/bin/env python3

import argparse
from typing import Iterable, Union
import pyonmttok
from collections import defaultdict
from itertools import takewhile, islice
import re
from io import TextIOWrapper, StringIO
import sys
import os
import codecs
import random
import json
from multipledispatch import dispatch

from utils.lemmatizer import Lemmatizer

random.seed(1234)
nl = "\n"
cr = "\r"
unk = "{unk,gl,gr}"


class PyonmttokWrapper:
    def __init__(
        self,
        model: str = None,
        add_in: bool = False,
        case_feature: bool = True,
        reserved_symbols: list = ["#", ":", "_", "\\", "|", "▁"],
    ):
        self.tokenizer = pyonmttok.Tokenizer(
            mode="aggressive", sp_model_path=model, case_feature=case_feature
        )
        self.add_in = add_in
        self.reserved_symbols = reserved_symbols

    @dispatch(str, str, str, float, float)
    def tokenize(
        self, src_path: str, tgt_path: str, constr_path: str, wskip: float, sskip: float
    ) -> None:
        self.tokenize(
            open(src_path, "r"), open(tgt_path, "w"), open(constr_path), wskip, sskip
        )

    @dispatch(str, str, str)
    def tokenize(self, src_path: str, tgt_path: str, constr_path: str) -> None:
        self.tokenize(open(src_path, "r"), open(tgt_path, "w"), open(constr_path, "w"))

    @dispatch(str, str)
    def tokenize(self, src_path: str, tgt_path: str) -> None:
        self.tokenize(open(src_path, "r"), open(tgt_path, "w"))

    @dispatch(TextIOWrapper, TextIOWrapper, TextIOWrapper, float, float)
    def tokenize(
        self,
        src: TextIOWrapper,
        tgt: TextIOWrapper,
        constr: TextIOWrapper,
        wskip: float,
        sskip: float,
    ) -> None:
        for s, c in zip(src, constr):
            tgt.write(self.tokenize(s, [eval(c)], wskip, sskip))
        src.close()
        tgt.close()
        constr.close()

    @dispatch(TextIOWrapper, TextIOWrapper, TextIOWrapper)
    def tokenize(
        self, src: TextIOWrapper, tgt: TextIOWrapper, constr: TextIOWrapper
    ) -> None:
        self.tokenize(src, tgt, constr, 0.0, 0.0)

    @dispatch(TextIOWrapper, TextIOWrapper)
    def tokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
        for s in src:
            tgt.write(self.tokenize(s))
        src.close()
        tgt.close()

    @dispatch(str)
    def tokenize(self, src: str) -> str:
        return self.tokenize([src])[0]

    @dispatch(list)
    def tokenize(self, src: list[str]) -> list[str]:
        """Tokenizes the given sentence(s)

        Args:
            src (list): list of input sentences

        Returns:
            output (list): tokenized sentences
        """

        def parse_bits(byte: str) -> int:
            """Parses the first 4 bits from the first byte (utf-8)

            Args:
                byte (str): string of byte

            Returns:
                cnt (int): number of bytes in sequence
            """
            bits, cnt = f"{int(byte, 16):0>8b}"[:4], 0
            while cnt < 4 and bits[cnt] != "0":
                cnt += 1
            return cnt if cnt > 0 else 1

        def byte_sequence2dec_sequence(byte_sequence: list[str]) -> str:
            """Translates hexadecimal byte sequence to a decimal sequence

            Args:
                byte_sequence (list): byte sequence

            Returns:
                decimal sequence (str): dec representation of the byte sequence
            """
            bytes_str = str(ord(bytearray.fromhex("".join(byte_sequence)).decode("utf8")))
            return f'{" ".join([f"<{c}>" for c in bytes_str])} <#>'

        def search_byte_pattern(txt: str) -> re.Match:
            """Searches for <0xDD> pattern in `txt`

            Args:
                txt (str): text

            Returns:
                bytes (re.Match): found byte sequence
            """

            return re.search(r"(?<=\<0[x])[\da-f]{2}(?=\>)", txt, flags=re.IGNORECASE)

        def get_join_factors(token: pyonmttok.Token) -> str:
            """Gets string representation of join factors

            Args:
                token (Token): token object

            Returns:
                factors (str): string representation of join factors
            """

            join_factors = "|gl+" if token.join_left else "|gl-"  # if join left
            join_factors += "|gr+" if token.join_right else "|gr-"  # if join right
            return join_factors

        def process_tokens(tokens: list[pyonmttok.Token]) -> Iterable[pyonmttok.Token]:
            tokens = iter(tokens)
            for token in tokens:
                byte = None

                # reserved symbols
                if token.surface in self.reserved_symbols:
                    byte = token.surface.encode()
                    hex = codecs.encode(byte, "hex").decode()
                    token.surface = f"<0x{hex}>"

                # alpha
                if token.surface.isalpha():
                    if len(token.surface) > 1:  # word/subword
                        token.features += (
                            "|ca"
                            if token.casing == pyonmttok.Casing.UPPERCASE
                            else "|ci"
                            if token.casing == pyonmttok.Casing.CAPITALIZED
                            else "|cn",
                        )
                    else:  # single character
                        token.features += (
                            "|scu"
                            if token.casing
                            in [pyonmttok.Casing.CAPITALIZED, pyonmttok.Casing.UPPERCASE,]
                            else "|scl",
                        )
                    # the beginning of word
                    if token.type in [
                        pyonmttok.TokenType.LEADING_SUBWORD,
                        pyonmttok.TokenType.WORD,
                    ]:
                        token.features += ("|wb",)
                    else:
                        token.features += ("|wbn",)
                    # |in factor
                    if self.add_in:
                        token.features += ("|in",)

                # numeric
                elif token.surface.isnumeric():
                    if token.type in [
                        pyonmttok.TokenType.LEADING_SUBWORD,
                        pyonmttok.TokenType.WORD,
                    ]:
                        token.features += ("|wb",)
                    else:
                        token.features += ("|wbn",)
                    if self.add_in:
                        token.features += ("|in",)

                # unicode (find first byte in byte sequence)
                elif byte := search_byte_pattern(token.surface):
                    token_sequence_length = parse_bits(
                        byte.group()
                    )  # number of tokens to be skipped
                    token.features = [unk, get_join_factors(token)]
                    token_sequence = [
                        token,
                        *islice(tokens, 0, token_sequence_length - 1),
                    ]
                    token.surface = byte_sequence2dec_sequence(
                        search_byte_pattern(token.surface).group()
                        for token in token_sequence
                    )

                # other
                else:
                    token.features = [
                        get_join_factors(token),
                        "|in" if self.add_in else "",
                    ]

                new_surface = (
                    token.surface.upper()
                    if token.surface.upper().lower() == token.surface
                    else token.surface
                )
                token.surface = (
                    f'{new_surface}{"".join(token.features)}'
                    if not byte
                    else f'{"".join(token.features)} {token.surface}'
                )
                yield token

        retval = []
        for sent in src:
            tokens = self.tokenizer.tokenize(sent, as_token_objects=True)
            tokens = process_tokens(tokens)
            tokenized_joined = " ".join(
                [
                    token.surface
                    for token in tokens
                    if token.surface and not search_byte_pattern(token.surface)
                ]
            )
            retval += (f'{tokenized_joined}{nl if sent.endswith((nl, cr)) else ""}',)
        return retval

    @dispatch(list, list, float, float)
    def tokenize(
        self,
        src: list[str],
        constraints: list[dict[tuple[int, int], list[str]]],
        wskip: float,
        sskip: float,
    ) -> list[str]:
        """Tokenizes the input with constraints

        Args:
            src (list): list of raw source sentences
            constraints (list): list of constraints of shape (range -> constraint)
            wskip (float): probability of a word skip
            sskip (float): probability of a sentence skip

        Returns:
            output (list): tokenized sentences
        """

        def generate_slices() -> Iterable[Iterable[str]]:
            """Generates slices for sentences.
            It is assumed that the only delimiter between words is whitespace.

            Returns:
                slices (Generator): iterable of slices
            """

            def _generate_slices() -> Iterable[str]:
                """Generates slices for a single sentence

                Returns:
                    slices (Generator): slices that were constructed based on constraints
                """

                prev_start_idx = 0
                for key, val in sorted(constr.items()):
                    # yield everything between constraint ranges
                    between = sent[prev_start_idx : key[0]]
                    if between.startswith(" "):
                        yield ["", ""]
                    if between.strip():
                        yield [between.strip(), ""]
                    if between.endswith(" ") and len(between) > 1:
                        yield ["", ""]
                    # yield constraint range
                    yield [sent[key[0] : key[1]], val]
                    prev_start_idx = key[1]

                # add last slice, even if its empty
                if sent[prev_start_idx:].startswith(" "):
                    prev_start_idx += 1
                    yield ["", ""]
                yield [sent[prev_start_idx:].strip(), ""]

            for sent, constr in zip(src, constraints):
                yield list(_generate_slices())

        def generate_tokenized() -> Iterable[str]:
            add_space = True
            byte_seq_pattern = r"^<[\d\#]>$"
            skip_sent = sskip > random.random()
            for s, c in zip(slice_tokenized, constr_tokenized):
                if not s:
                    add_space = True
                    continue
                # if constraint -> join both
                if c and not skip_sent and wskip < random.random():
                    s = " ".join(
                        [
                            *[
                                f"{s_sw}|t1"
                                if not re.search(byte_seq_pattern, s_sw)
                                else s_sw
                                for s_sw in s.split()
                            ],
                            *[
                                f"{c_sw}|t2"
                                if not re.search(byte_seq_pattern, c_sw)
                                else c_sw
                                for c_sw in c.split()
                            ],
                        ]
                    )
                else:
                    s = " ".join(
                        [
                            f"{s_sw}|t0"
                            if not re.search(byte_seq_pattern, s_sw)
                            else s_sw
                            for s_sw in s.split()
                        ]
                    )

                first, *others = s.split(" ", 1)
                if add_space:
                    first = first.replace("|gl+", "|gl-")
                    first = first.replace("|wbn", "|wb")
                    add_space = False
                else:
                    first = first.replace("|gl-", "|gl+")
                    first = first.replace("|wb", "|wbn")
                yield " ".join([first, *others])

        if len(src) != len(constraints):
            # since we are iterating over constarints, we need to add empty constraints,
            # so their count will correspond to that of input sentences
            while len(src) != len(constraints):
                constraints += ({},)

        retval = []
        src_slices = generate_slices()
        for slice in src_slices:
            # transpose
            sliced_src, sliced_constr = (list(s) for s in list(zip(*slice)))
            slice_tokenized = self.tokenize(sliced_src)
            constr_tokenized = self.tokenize(sliced_constr)
            tokenized_joined = " ".join(generate_tokenized())
            retval += (f'{tokenized_joined}{nl if src[0].endswith((nl, cr)) else ""}',)
        return retval

    @dispatch(list, list)
    def tokenize(
        self, src: list[str], constraints: list[dict[tuple[int, int], list[str]]]
    ) -> list[str]:
        return self.tokenize(src, constraints, 0.0, 0.0)

    @dispatch(str, str)
    def detokenize(self, src_path: str, tgt_path: str) -> None:
        """Detokenizes the source file

        Args:
            src_path (str): path to source file
            tgt_path (str): path to target file
        """
        self.detokenize(open(src_path, "r"), open(tgt_path, "w"))

    @dispatch(TextIOWrapper, TextIOWrapper)
    def detokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
        for s in src:
            tgt.write(self.detokenize(s))
        src.close()
        tgt.close()

    @dispatch(str)
    def detokenize(self, src: str) -> str:
        return self.detokenize([src])[0]

    @dispatch(list)
    def detokenize(self, src: list[str]) -> list[str]:
        """Detokenizes sentence(s)

        Args:
            src (list): list of input sentences

        Returns:
            output (list): detokenized sentences
        """

        def extract_subword_n_factors(token: str) -> tuple[str, list[str]]:
            try:
                subword, factors = token.split("|", 1)
            except ValueError:
                subword, factors = token, "|"
            return subword, factors.split("|")

        def find_any(factors: list, *factors2find: list[str]) -> bool:
            return any(factor in factors2find for factor in factors)

        def assign_join(token: pyonmttok.Token, factors: list[str]) -> None:
            token.join_left = True if "gl+" in factors else False
            token.join_right = True if "gr+" in factors else False

        def process_tokens(tokens: list[str]) -> list[pyonmttok.Token]:
            tokens = iter(tokens)
            for token in tokens:
                subword, factors = extract_subword_n_factors(token)
                new_token = pyonmttok.Token()

                # byte sequence
                if re.search(unk, token):
                    byte_sequence_factors = extract_subword_n_factors(token)[1]
                    byte_sequence = "".join(
                        re.search(r"(?<=<)\d(?=>)", next_token).group(0)
                        for next_token in takewhile(lambda t: t != "<#>", tokens)
                    )
                    try:
                        new_token.surface = chr(int(byte_sequence))
                    # invalid byte sequence
                    except OverflowError:
                        new_token.surface = ""
                    else:
                        new_token.type = pyonmttok.TokenType.WORD
                        new_token.spacer = True
                        new_token.casing = pyonmttok.Casing.NONE
                        # make it empty space if byte sequence is newline/carriage return
                        new_token.surface = (
                            " " if new_token.surface in [nl, cr] else new_token.surface
                        )
                    assign_join(new_token, byte_sequence_factors)

                elif find_any(factors, "wbn", "wb"):
                    # assign casing and surface
                    if find_any(factors, "scu", "ca"):
                        new_token.casing = pyonmttok.Casing.UPPERCASE
                        new_token.surface = subword
                    elif "ci" in factors:
                        new_token.casing = pyonmttok.Casing.CAPITALIZED
                        new_token.surface = subword.lower().capitalize()
                    elif find_any(factors, "scl", "cn"):
                        new_token.casing = pyonmttok.Casing.LOWERCASE
                        new_token.surface = subword.lower()
                    else:
                        new_token.casing = pyonmttok.Casing.NONE
                        new_token.surface = subword
                    # assign type, join_left and spacer
                    if "wbn" in factors:
                        new_token.type = pyonmttok.TokenType.TRAILING_SUBWORD
                        new_token.join_left = True
                        new_token.spacer = False
                    else:
                        new_token.type = pyonmttok.TokenType.WORD
                        new_token.join_left = False
                        new_token.spacer = True

                # punctuation, emoji
                else:
                    new_token.surface = subword
                    new_token.type = pyonmttok.TokenType.WORD
                    new_token.spacer = True
                    new_token.casing = pyonmttok.Casing.NONE
                    assign_join(new_token, factors)

                yield new_token

        retval = []
        for sent in src:
            tokens = sent.split()
            tokens = list(process_tokens(tokens))
            detokenized_joined = self.tokenizer.detokenize(tokens)
            # add newline if there is one in the source sentence
            retval += (f'{detokenized_joined}{nl if sent.endswith((nl, cr)) else ""}',)

        return retval

    def generate_constraints(
        self,
        src_path: str,
        tgt_path: str,
        vocab_path: str,
        constraints_file: str = "",
        noisify: bool = True,
        use_lemmatization: bool = True,
        distance_threshold: float = 0.4,
        src_lang: str = "",
        tgt_lang: str = "",
        liblemm_path: str = "lib",
    ) -> Union[StringIO, list[dict[tuple[int, int], str]]]:
        """Generates the list of constraints

        Args:
            src_path (str): path to the source file
            tgt_path (str): path to the target file
            vocab_path (str): path to the bilingual dictionary
            constraints_file (str): file path to a file with resulting constraints
            noisify (bool): noisify the substitutions
            use_lemmatization (bool): add lemmatized versions to the dictionary and
              source/target sentences
            distance_threshold (float): distance threshold
              (1.0 == length of target sentence) between matched words
            liblemm_path (str): path to the folder with vocabularies
              and language codes (lingea)
            src_lang (str): iso code of the source language
            tgt_lang (str): iso code of the target language

        Returns:
            constraints (dict|TextIO): list of ranges with corresponding words
        """

        lemmatize = False
        if use_lemmatization and os.path.exists(liblemm_path):
            if not all((src_lang, tgt_lang)):
                src_lang, tgt_lang = vocab_path.rsplit(".", 1)[1].split(
                    ("-", "_", ",", ":")
                )
            src_vocab_path = os.path.join(liblemm_path, f"lgmf_{src_lang}.lex")
            tgt_vocab_path = os.path.join(liblemm_path, f"lgmf_{tgt_lang}.lex")
            liblemm_iso_codes = os.path.join(liblemm_path, "liblemm_iso_codes.so")
            try:
                src_lemmatizer = Lemmatizer(
                    src_lang, vocab=src_vocab_path, encoding="il2", path=liblemm_iso_codes
                )
                tgt_lemmatizer = Lemmatizer(
                    tgt_lang, vocab=tgt_vocab_path, encoding="il2", path=liblemm_iso_codes
                )
            except:
                pass
            else:
                lemmatize = True

        vocab = defaultdict(set)

        # read the vocab
        with open(vocab_path, "r") as vocab_lines:
            for line in vocab_lines:
                src_word, tgt_word = (word for word in line.rstrip().split("\t"))
                vocab[src_word].add(tgt_word)
                if lemmatize:
                    for src_word_lemma in src_lemmatizer.lemmatize(src_word):
                        vocab[src_word_lemma] = vocab[src_word_lemma].union(
                            tgt_lemmatizer.lemmatize(tgt_word)
                        )

        retval = []

        def add2retval(
            idx: int, tgt_word: str, spans: tuple[tuple[int, int], tuple[int, int]]
        ) -> None:
            if noisify:
                # Esentially the same noisification as the one in the MLM
                # 10% - do nothing
                # 10% - random token from vocab
                # 80% - normal substitution
                dice_roll = random.uniform(0, 1)
                if dice_roll < 0.1:
                    return
                elif dice_roll < 0.2:
                    random_values = random.choice(list(vocab.values()))
                    retval[idx][spans] = random.choice(list(random_values))
                else:
                    retval[idx][spans] = tgt_word
            else:
                retval[idx][spans] = tgt_word
            used_src_words[src_word] += 1
            used_tgt_words[tgt_word] += 1

        def find_spans():
            word_boundaries = r"(?<![\w_]){word}(?![\w_])"
            src_span = list(re.finditer(word_boundaries.format(word=src_word), src_sent))[
                used_src_words[src_word]
            ].span()
            try:
                tgt_span = list(
                    re.finditer(word_boundaries.format(word=tgt_word), tgt_sent)
                )[used_tgt_words[tgt_word]].span()
            except IndexError:
                used_tgt_words[tgt_word] = -1
                return None
            else:
                return src_span, tgt_span

        # generate constraints
        with open(src_path, "r") as src, open(tgt_path, "r") as tgt:
            tokenizer = pyonmttok.Tokenizer(mode="aggressive")
            for idx, (src_sent, tgt_sent) in enumerate(zip(src, tgt)):
                retval += ({},)
                # used_{src,tgt}_words are dictionaries that are used
                # for the storage of current index of some word in src/tgt sentence.
                # it is useful for the case when we have non-singular number
                # of some word in a sentence
                used_src_words = defaultdict(int)
                used_tgt_words = defaultdict(int)
                tok_src_sent = tokenizer.tokenize(src_sent, as_token_objects=False)[0]
                tok_tgt_sent = tokenizer.tokenize(tgt_sent, as_token_objects=False)[0]
                for src_word in tok_src_sent:
                    # 0-th index will contain non-lemmatized mappings from the vocab
                    tgt_vocab_words = [{None}]
                    # non-lemmatized word is present in the vocabulary
                    if src_word in vocab:
                        tgt_vocab_words[0] = vocab[src_word]
                    # one of lemmas is present in the vocabulary
                    elif lemmatize:
                        # for some reason, our lemmatizer returns multiple lemmas for
                        # a single word
                        for src_word_lemma in src_lemmatizer.lemmatize(src_word):
                            # src_word_lemma = src_word_lemma.decode()
                            if src_word_lemma in vocab:
                                tgt_vocab_words += (vocab[src_word_lemma],)

                    if not tgt_vocab_words:
                        continue

                    for tgt_word in tok_tgt_sent:
                        spans = ()
                        # non-lemmatized
                        if (
                            tgt_word in tgt_vocab_words[0]
                            and used_tgt_words[tgt_word] != -1
                            and (spans := find_spans())
                        ):
                            if (
                                (spans[1][0] - spans[0][0]) / len(tgt_sent)
                                > distance_threshold
                            ):
                                continue
                            add2retval(idx, tgt_word, spans)
                            break
                        # lemmatized
                        elif lemmatize:
                            if (
                                any(
                                    tgt_vocab_lemmas.intersection(
                                        lemma
                                        for lemma in tgt_lemmatizer.lemmatize(tgt_word)
                                    )
                                    for tgt_vocab_lemmas in tgt_vocab_words[1:]
                                )
                                and used_tgt_words[tgt_word] != -1
                                and (spans := find_spans())
                            ):
                                if (
                                    (spans[1][0] - spans[0][0]) / len(tgt_sent)
                                    > distance_threshold
                                ):
                                    continue
                                add2retval(idx, tgt_word, spans)
                                break

        if constraints_file:
            with open(constraints_file, "w") as constraints:
                constraints.write('\n'.join(str(line) for line in retval))
        return retval


def parse_args():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--tokenize", action="store_true", help="Use this flag to tokenize a file"
    )
    action.add_argument(
        "--detokenize", action="store_true", help="Use this flag to detokenize a file"
    )
    action.add_argument(
        "--generate", action="store_true", help="Use this flag to generate constraints"
    )
    parser.add_argument(
        "-s",
        "--src",
        default=sys.stdin,
        type=str,
        help="Either a path to source file or string to be processed",
    )
    parser.add_argument(
        "-t", "--tgt", default=sys.stdout, type=str, help="Path to target file"
    )
    parser.add_argument(
        "-c", "--constraints", default="", type=str, help="Path to constraints file"
    )
    parser.add_argument(
        "--wskip",
        default=0.0,
        type=float,
        help="Word skip probability, for training only",
    )
    parser.add_argument(
        "--sskip",
        default=0.0,
        type=float,
        help="Sentence skip probability, for training only",
    )
    parser.add_argument("-m", "--model", default=None, type=str, help="Path to SP model")
    parser.add_argument("--add_in", action="store_true", default=False)
    parser.add_argument(
        "--no_case_feature", action="store_false", dest="case_feature", default=True
    )
    parser.add_argument(
        "--constr_vocab",
        default="",
        type=str,
        help="Path to the constraints vocabulary (bilingual word mappings)",
    )
    parser.add_argument(
        "--constr_out",
        default=sys.stdout,
        type=str,
        help="Path to the output constraints file",
    )
    return parser.parse_args()


# cli
if __name__ == "__main__":
    args = parse_args()
    tokenizer = PyonmttokWrapper(
        model=args.model, add_in=args.add_in, case_feature=args.case_feature
    )
    if args.tokenize:
        tokenizer.tokenize(
            args.src, args.tgt, args.constraints, args.wskip, args.sskip
        ) if args.constraints else tokenizer.tokenize(args.src, args.tgt)
    elif args.generate:
        if args.tgt is sys.stdout:
            raise argparse.ArgumentError(
                "Target file cannot be stdout when generating constraints"
            )
        tokenizer.generate_constraints(
            args.src, args.tgt, args.constr_vocab, False, args.constr_out, True, True
        )
    else:
        tokenizer.detokenize(args.src, args.tgt)
