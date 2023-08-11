#!/usr/bin/env python3

import argparse
from typing import Iterable, Optional, Iterator
import pyonmttok
from collections import defaultdict
from itertools import takewhile, islice, tee
import re
from io import TextIOWrapper
import sys
from tqdm import tqdm
import os
import codecs
import random
from multipledispatch import dispatch
from utils.lemmatizer import Lemmatizer

random.seed(1234)
nl = "\n"
cr = "\r"
unk = "{unk,gl,gr}"


def count_lines(i: Iterator) -> int:
    cnt = 0
    for _ in i:
        cnt += 1
    return cnt


def wrap_tqdm(to_be_wrapped: Iterable, desc: str, n_lines: int) -> Iterable:
    if __name__ == "__main__":
        return tqdm(to_be_wrapped, desc=desc, total=n_lines)
    else:
        return to_be_wrapped


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

    # Tokenize overloads:
    # tokenize(src_path: str, tgt_path: str, constraints_path: str) -> None
    # tokenize(src: TextIO, tgt: TextIO, constraints: TextIO) -> None
    # tokenize(src_path: str, tgt_path: str) -> None
    # tokenize(src: TextIO, tgt: TextIO) -> None
    # tokenize(src: str) -> str
    # tokenize(src: str, constraints: dict) -> str
    # tokenize(src: list) -> list
    # tokenize(src: list, tgt: list, constraints: list) -> tuple[list, list]
    # tokenize(src: list constraints: list) -> list

    # Detokenize overloads:
    # detokenize(src_path: str, tgt_path: str) -> None
    # detokenize(src: TextIO, tgt: TextIO) -> None
    # detokenize(src: str) -> str
    # detokenize(src: list) -> list

    @dispatch(str, str, str)
    def tokenize(self, src_path: str, tgt_path: str, constraints_path: str) -> None:
        self.tokenize(
            open(src_path, "r"), open(tgt_path, "w"), open(constraints_path, "r")
        )

    @dispatch(TextIOWrapper, TextIOWrapper, TextIOWrapper)
    def tokenize(
        self, src: TextIOWrapper, tgt: TextIOWrapper, constraints: TextIOWrapper
    ) -> None:
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)
        for src_sent, constraint in wrap_tqdm(
            to_be_wrapped=zip(src_iter, constraints), desc=f"Tokenizing {src.name}", n_lines=n_lines
        ):
            tgt.write(self.tokenize(src_sent, eval(constraint)))
        src.close()
        tgt.close()
        constraints.close()

    @dispatch(str, str)
    def tokenize(self, src_path: str, tgt_path: str) -> None:
        self.tokenize(open(src_path, "r"), open(tgt_path, "w"))

    @dispatch(TextIOWrapper, TextIOWrapper)
    def tokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)
        for src_sent in wrap_tqdm(
            to_be_wrapped=src_iter, desc=f"Tokenizing {src.name}", n_lines=n_lines
        ):
            tgt.write(self.tokenize(src_sent))
        src.close()
        tgt.close()

    @dispatch(str)
    def tokenize(self, src: str) -> str:
        return self.tokenize([src])[0]

    @dispatch(str, dict)
    def tokenize(self, src: str, constraints: dict[tuple[int, int], str]) -> str:
        return self.tokenize([src], [constraints])[0]

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

    @dispatch(list, list, list)
    def tokenize(
        self,
        src: list[str],
        tgt: list[str],
        constraints: list[dict[tuple[tuple[int, int], tuple[int, int]], str]],
    ) -> tuple[list[str], list[str]]:
        def generate_tuples(
            constraints: list[dict[tuple[tuple[int, int], tuple[int, int]], str]],
            idx: int,
        ) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
            for constraint in constraints:
                yield tuple({c[0][idx]: c[1]} for c in constraint.items())

        src_constraints = generate_tuples(constraints, 0)
        # tgt_constraints = generate_tuples(constraints, 1)  # if we should use it in tgt?
        return self.tokenize(src, src_constraints), self.tokenize(tgt)

    @dispatch(list, list)
    def tokenize(
        self, src: list[str], constraints: list[dict[tuple[int, int], str]]
    ) -> list[str]:
        """Tokenizes the input with constraints

        Args:
            src (list): list of raw source sentences
            constraints (list): list of constraints of shape (range -> constraint)

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
            for s, c in zip(slice_tokenized, constr_tokenized):
                if not s:
                    add_space = True
                    continue
                # if constraint -> join both
                if c:
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
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)
        for src_sent in wrap_tqdm(to_be_wrapped=src_iter, desc=f"Detokenizing {src.name}", n_lines=n_lines):
            tgt.write(self.detokenize(src_sent))
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
        constraints_path: str,
        return_list: bool = True,
        wskip: float = 0.0,
        wrand: float = 0.0,
        sskip: float = 0.0,
        use_lemmatization: bool = True,
        distance_limit: int = 80,
        distance_threshold: float = 0.4,
        src_lang: str = "",
        tgt_lang: str = "",
        liblemm_path: str = "lib",
    ) -> Optional[list[dict[tuple[int, int], str]]]:
        """Generates the list of constraints

        Args:
            src_path (str): path to the source file
            tgt_path (str): path to the target file
            vocab_path (str): path to the bilingual dictionary
            constraints_path (str): file path to a file with resulting constraints
            return_list (bool): if we want to return a list of constraints
            wskip (float): probability of word substitution skip
            wrand (float): probability of word substitution randomization
            sskip (float): probability of sentence skip
              (leave it with the empty dict of constraints)
            distance_limit (int): if length of the target sentence is
              more than distance_limit, distance threshold test will be applied
            distance_threshold (float): distance threshold
              (1.0 == length of target sentence) between matched words
            use_lemmatization (bool): add lemmatized versions to the dictionary and
              source/target sentences
            src_lang (str): iso code of the source language
            tgt_lang (str): iso code of the target language
            liblemm_path (str): path to the folder with vocabularies
              and language codes (lingea)
        Returns:
            constraints (list): list of ranges with corresponding words
        """

        lemmatize = False
        if use_lemmatization and os.path.exists(liblemm_path):
            if not all((src_lang, tgt_lang)):
                vocab_langs = vocab_path.rsplit(".", 1)[1]
                src_lang = vocab_langs[:2]
                tgt_lang = vocab_langs[-2:]
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
                        src_word_lemma = src_word_lemma.decode()
                        vocab[src_word_lemma] = vocab[src_word_lemma].union(
                            tgt_word_lemma.decode()
                            for tgt_word_lemma in tgt_lemmatizer.lemmatize(tgt_word)
                        )

        retval = []

        def find_span(
            word: str, sent: str, used_words: dict[str, int]
        ) -> Optional[tuple[int, int]]:
            if used_words[word] == -1:
                return None
            try:
                span = list(
                    re.finditer(
                        r"(?<![\w_]){word}(?![\w_])".format(word=re.escape(word)), sent
                    )
                )[used_words[word]].span()
            except IndexError:
                used_words[word] = -1
                return None
            return span

        def process_constr(
            tgt_word: str,
            tgt_sent_length: int,
            src_span: tuple[int, int],
            tgt_span: tuple[int, int],
        ) -> Optional[str]:
            if (
                tgt_sent_length > distance_limit
                and abs(tgt_span[0] - src_span[0]) / tgt_sent_length > distance_threshold
            ):
                return None
            if wskip or wrand:
                # Esentially the same noisification as the one in the MLM
                # wskip (10% by default) - do nothing
                # wrand (10% by default) - random token from the vocab
                # other - normal substitution
                dice_roll = random.uniform(0.0, 1.0)
                if wskip and dice_roll < wskip:
                    return None
                elif wrand and dice_roll < wskip + wrand:
                    random_values = random.choice(list(vocab.values()))
                    return random.choice(list(random_values))
            return tgt_word

        # generate constraints
        constr_out = (
            open(constraints_path, "w")
            if constraints_path is not sys.stdout
            else sys.stdout
            if constraints_path
            else None
        )

        with open(src_path, "r") as src, open(tgt_path, "r") as tgt:
            src_iter, src_iter_copy = tee(iter(src))
            n_lines = count_lines(src_iter_copy)
            tokenizer = pyonmttok.Tokenizer(mode="aggressive")
            for src_sent, tgt_sent in wrap_tqdm(
                to_be_wrapped=zip(src_iter, tgt),
                desc=f"Generating constraints for {src.name}",
                n_lines=n_lines
            ):
                rv = {}
                if sskip and random.uniform(0.0, 1.0) < sskip:
                    if return_list:
                        retval += (rv,)
                    else:
                        constr_out.write(
                            f"{str(rv)}{nl if nl in src_sent or nl in tgt_sent else ''}"
                        )
                    continue
                # used_{src,tgt}_words are dictionaries that are used
                # for the storage of current index of some word in src/tgt sentence.
                # it is useful for the case when we have non-singular number
                # of some word in a sentence
                src_used_words = defaultdict(int)
                tgt_used_words = defaultdict(int)
                tok_src_sent = tokenizer.tokenize(src_sent, as_token_objects=False)[0]
                tok_tgt_sent = tokenizer.tokenize(tgt_sent, as_token_objects=False)[0]
                for src_word in tok_src_sent:
                    # set at the 0-th index will contain
                    # non-lemmatized mappings from the vocab
                    tgt_vocab_words = [set()]
                    # non-lemmatized word is present in the vocabulary
                    if src_word in vocab:
                        tgt_vocab_words[0] = vocab[src_word]
                    # if any of lemmas are present in the vocabulary
                    elif lemmatize:
                        # for some reason, our lemmatizer returns multiple lemmas for
                        # a single word
                        for src_word_lemma in src_lemmatizer.lemmatize(src_word):
                            if (src_word_lemma := src_word_lemma.decode()) in vocab:
                                tgt_vocab_words += (vocab[src_word_lemma],)
                    # check if all mapping sets are empty
                    if not any(tgt_vocab_words):
                        continue
                    # can't find word/no word to map
                    if not (src_span := find_span(src_word, src_sent, src_used_words)):
                        continue

                    for tgt_word in tok_tgt_sent:
                        # get span
                        if not (
                            tgt_span := find_span(tgt_word, tgt_sent, tgt_used_words)
                        ):
                            continue
                        if tgt_word in tgt_vocab_words[0] or (
                            lemmatize
                            and any(
                                tgt_vocab_lemmas.intersection(
                                    lemma.decode()
                                    for lemma in tgt_lemmatizer.lemmatize(tgt_word)
                                )
                                for tgt_vocab_lemmas in tgt_vocab_words[1:]
                            )
                        ):
                            if constraint := process_constr(
                                tgt_word, len(tgt_sent), src_span, tgt_span
                            ):
                                rv[src_span] = constraint
                # add constraints to the result (either final list or out file)
                if return_list:
                    retval += (rv,)
                else:
                    constr_out.write(
                        f"{str(rv)}{nl if nl in src_sent or nl in tgt_sent else ''}"
                    )
        if constr_out and constr_out is not sys.stdout:
            constr_out.close()
        if return_list:
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
        dest="src_path",
        default=sys.stdin,
        type=str,
        help="Either a path to source file or string to be processed",
    )
    parser.add_argument(
        "-t",
        "--tgt",
        dest="tgt_path",
        default=sys.stdout,
        type=str,
        help="Path to target file",
    )
    parser.add_argument(
        "-c",
        "--constr",
        dest="constraints_path",
        default='',
        type=str,
        help="Path to the constraints file",
    )
    parser.add_argument(
        "--src_lang",
        default="",
        type=str,
        help="Source language (this parameter is required for the constraints generation only)",
    )
    parser.add_argument(
        "--tgt_lang",
        default="",
        type=str,
        help="Target language (this parameter is required for the constraints generation only)",
    )
    parser.add_argument(
        "--wskip",
        default=0.0,
        type=float,
        help="Word skip probability, for training data modification only",
    )
    parser.add_argument(
        "--wrand",
        default=0.0,
        type=float,
        help="Random substitution probability, for training data modification only",
    )
    parser.add_argument(
        "--sskip",
        default=0.0,
        type=float,
        help="Sentence skip probability, for training data modification only",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_path",
        default=None,
        type=str,
        help="Path to SP model",
    )
    parser.add_argument("--add_in", action="store_true", default=False)
    parser.add_argument(
        "--no_case_feature", action="store_false", dest="case_feature", default=True
    )
    parser.add_argument(
        "--constr_vocab",
        dest="constraints_vocab_path",
        default="",
        type=str,
        help="Path to the constraints vocabulary (bilingual word mappings)",
    )
    parser.add_argument(
        "--distance_threshold", default=0.4, type=float, help="Distance threshold m"
    )
    parser.add_argument(
        "--distance_limit",
        default=80,
        type=int,
        help="Distance limit is used to define the ",
    )
    return parser.parse_args()


# cli
if __name__ == "__main__":
    args = parse_args()
    tokenizer = PyonmttokWrapper(
        model=args.model_path, add_in=args.add_in, case_feature=args.case_feature
    )
    # convert i/o to TextIOWrappers
    if args.generate:
        if not args.constraints_path:
            args.constraints_path = sys.stdout
        if args.tgt_path is sys.stdout:
            raise argparse.ArgumentError(
                "Target file cannot be stdout when generating constraints"
            )
        tokenizer.generate_constraints(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            vocab_path=args.constraints_vocab_path,
            constraints_path=args.constraints_path,
            return_list=False,
            wskip=args.wskip,
            wrand=args.wrand,
            sskip=args.sskip,
            use_lemmatization=True,
            distance_limit=args.distance_limit,
            distance_threshold=args.distance_threshold,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            liblemm_path="lib",
        )
    else:
        with open(
            args.src_path, "r"
        ) if args.src_path is not sys.stdin else sys.stdin as src, open(
            args.tgt_path, "w"
        ) if args.tgt_path is not sys.stdout else sys.stdout as tgt, open(
            args.constraints_path, "r"
        ) if args.constraints_path is not sys.stdout else sys.stdin as constraints:
            if args.tokenize:
                tokenizer.tokenize(
                    src, tgt, constraints
                ) if args.constraints_path else tokenizer.tokenize(src, tgt)
            else:
                tokenizer.detokenize(src, tgt)
