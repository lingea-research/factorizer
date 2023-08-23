#!/usr/bin/env python3

import argparse
from typing import Iterable, Optional, Iterator
from pyonmttok import Token, Tokenizer, TokenType, Casing, SentencePieceLearner
from collections import defaultdict
from itertools import takewhile, islice, tee
import re
from io import TextIOWrapper
import sys
import copy
from tqdm import tqdm
import os
import codecs
import random
from multipledispatch import dispatch
from .utils.lemmatizer import Lemmatizer

random.seed(1234)
nl = "\n"
cr = "\r"
unk = "{unk,gl,gr}"
silent = False


def count_lines(i: Iterator) -> int:
    cnt = 0
    for _ in i:
        cnt += 1
    return cnt


def wrap_tqdm(to_be_wrapped: Iterable, desc: str, n_lines: int) -> Iterable:
    if __name__ == "__main__" and not silent:
        return tqdm(to_be_wrapped, desc=desc, total=n_lines)
    else:
        return to_be_wrapped


class FactoredTokenizer:
    def __init__(
        self,
        model_path: str = None,
        add_in: bool = False,
        add_constr: bool = False,
        case_feature: bool = True,
        reserved_symbols: list = ["#", ":", "_", "\\", "|", "▁"],
    ):
        self.model_path = model_path
        self.add_in = add_in
        self.add_constr = add_constr
        self.case_feature = case_feature
        try:
            self.tokenizer = Tokenizer(
                mode="aggressive", sp_model_path=model_path, case_feature=case_feature
            )
        except ValueError:  # path does not exist (spm_train is used)
            self.tokenizer = None
        self.add_in = add_in
        self.reserved_symbols = reserved_symbols

    @dispatch(str, str, str)
    def tokenize(self, src_path: str, tgt_path: str, constraints_path: str) -> None:
        with (
            open(src_path, "r") as src,
            open(tgt_path, "w") as tgt,
            open(constraints_path, "r") as constraints,
        ):
            self.tokenize(src, tgt, constraints)

    @dispatch(TextIOWrapper, TextIOWrapper, TextIOWrapper, int)
    def tokenize(
        self,
        src: TextIOWrapper,
        tgt: TextIOWrapper,
        constraints: TextIOWrapper,
        multispan_index=-1,
    ) -> None:
        if multispan_index > -1:
            constraints = list(self.generate_tuples(constraints, multispan_index))
        self.tokenize(src, tgt, constraints)

    @dispatch(TextIOWrapper, TextIOWrapper, (list, TextIOWrapper))
    def tokenize(
        self, src: TextIOWrapper, tgt: TextIOWrapper, constraints: TextIOWrapper
    ) -> None:
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)

        for src_sent, constraint in wrap_tqdm(
            to_be_wrapped=zip(src_iter, constraints),
            desc=f"Tokenizing {src.name}",
            n_lines=n_lines,
        ):
            if type(constraint) != dict:
                constraint = eval(constraint)
            tgt.write(self.tokenize(src_sent, constraint))

    @dispatch(str, str)
    def tokenize(self, src_path: str, tgt_path: str) -> None:
        with open(src_path, "r") as src, open(tgt_path, "w") as tgt:
            self.tokenize(src, tgt)

    @dispatch(TextIOWrapper, TextIOWrapper)
    def tokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)
        for src_sent in wrap_tqdm(
            to_be_wrapped=src_iter, desc=f"Tokenizing {src.name}", n_lines=n_lines
        ):
            tgt.write(self.tokenize(src_sent))

    @dispatch(str)
    def tokenize(self, src: str) -> str:
        def search_byte_pattern(txt: str) -> re.Match:
            """Searches for <0xDD> pattern in `txt`

            Args:
                txt (str): text

            Returns:
                bytes (re.Match): found byte sequence
            """

            return re.search(r"(?<=\<0[x])[\da-f]{2}(?=\>)", txt, flags=re.IGNORECASE)

        def process_tokens(tokens: list[Token]) -> Iterable[Token]:
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
                bytes_str = str(
                    ord(bytearray.fromhex("".join(byte_sequence)).decode("utf8"))
                )
                return f'{" ".join([f"<{c}>" for c in bytes_str])} <#>'

            def get_join_factors(token: Token) -> str:
                """Gets string representation of join factors

                Args:
                    token (Token): token object

                Returns:
                    factors (str): string representation of join factors
                """

                join_factors = "|gl+" if token.join_left else "|gl-"  # if join left
                join_factors += "|gr+" if token.join_right else "|gr-"  # if join right
                return join_factors

            tokens = iter(tokens)
            join_left = False
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
                            if token.casing == Casing.UPPERCASE
                            else "|ci"
                            if token.casing == Casing.CAPITALIZED
                            else "|cn",
                        )
                    else:  # single character
                        token.features += (
                            "|scu"
                            if token.casing
                            in [
                                Casing.CAPITALIZED,
                                Casing.UPPERCASE,
                            ]
                            else "|scl",
                        )
                    # the beginning of word
                    if (
                        token.type
                        in [
                            TokenType.LEADING_SUBWORD,
                            TokenType.WORD,
                        ]
                        and not join_left
                    ):
                        token.features += ("|wb",)
                    else:
                        token.features += ("|wbn",)
                    # |in factor
                    if self.add_in:
                        token.features += ("|in",)
                    if self.add_constr:
                        token.features += ("|t0",)

                # numeric
                elif token.surface.isnumeric():
                    if token.type in [
                        TokenType.LEADING_SUBWORD,
                        TokenType.WORD,
                    ]:
                        token.features += ("|wb",)
                    else:
                        token.features += ("|wbn",)
                    if self.add_in:
                        token.features += ("|in",)
                    if self.add_constr:
                        token.features += ("|t0",)

                # unicode (find first byte in byte sequence)
                elif byte := search_byte_pattern(token.surface):
                    token_sequence_length = parse_bits(
                        byte.group()
                    )  # number of tokens to be skipped
                    token.features = [
                        unk,
                        get_join_factors(token),
                        "|t0" if self.add_constr else "",
                    ]
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
                        "|t0" if self.add_constr else "",
                    ]

                new_surface = (
                    token.surface.upper()
                    if token.surface.upper().lower() == token.surface
                    else token.surface
                )

                # current token has join_right == True,
                # so the next token will have join_left = True
                join_left = False
                if token.join_right == True:
                    join_left = True

                token.surface = (
                    f'{new_surface}{"".join(token.features)}'
                    if not byte
                    else f'{"".join(token.features)} {token.surface}'
                )
                yield token

        tokens = self.tokenizer.tokenize(src, as_token_objects=True)
        tokens = process_tokens(tokens)
        tokenized_joined = " ".join(
            [
                token.surface
                for token in tokens
                if token.surface and not search_byte_pattern(token.surface)
            ]
        )
        return f'{tokenized_joined}{nl if src.endswith((nl, cr)) else ""}'

    @dispatch(str, dict)
    def tokenize(self, src: str, constraints: dict[tuple[int, int], str]) -> str:
        def generate_slices(
            sent: str, constr: dict[tuple[int, int], str]
        ) -> Iterable[tuple[str, str]]:
            """Generates slices for a single sentence

            Returns:
                slices (Generator): slices that were constructed based on constraints
            """

            prev_start_idx = 0
            for key, val in sorted(constr.items()):
                # yield everything between constraint ranges
                between = sent[prev_start_idx : key[0]]
                if between.startswith(" "):
                    yield ("", "")
                if between_stripped := between.strip():
                    yield (between_stripped, "")
                if between.endswith(" ") and len(between) > 1:
                    yield ("", "")
                # yield constraint range
                yield (sent[key[0] : key[1]], val)
                prev_start_idx = key[1]

            # add last slice, even if it's empty
            if sent[prev_start_idx:].startswith(" "):
                prev_start_idx += 1
                yield ("", "")
            yield (sent[prev_start_idx:].strip(), "")

        def generate_tokenized(
            src_slice_tokenized: list, constr_slice_tokenized: list
        ) -> Iterable[str]:
            add_space = True
            byteseq_byte_pattern = r"^<[\d\#]>$"

            def assign_factors(s: list, constr_factor: str = "") -> Iterable:
                return [
                    f"{sw}|{constr_factor}"
                    if not re.search(byteseq_byte_pattern, sw)
                    else sw
                    for sw in s.split()
                ]

            for s, c in zip(src_slice_tokenized, constr_slice_tokenized):
                if not s:
                    add_space = True
                    continue
                # if constraint -> join both
                if s == c:
                    s = " ".join(assign_factors(s, "t2"))
                elif c:
                    s = " ".join(
                        [
                            *assign_factors(s, "t1"),
                            *assign_factors(c, "t2"),
                        ]
                    )
                else:
                    s = " ".join(assign_factors(s, "t0"))

                first, *others = s.split(" ", 1)
                if add_space:
                    first = first.replace("|gl+", "|gl-")
                    first = first.replace("|wbn", "|wb")
                    add_space = False
                else:
                    first = first.replace("|gl-", "|gl+")
                    first = first.replace("|wb", "|wbn")
                yield " ".join([first, *others])

        slices = generate_slices(src, constraints)
        src_slice, constr_slice = (list(s) for s in list(zip(*slices)))
        src_slice_tokenized = self.tokenize(src_slice)
        constr_sliced_tokenized = self.tokenize(constr_slice)
        tokenized_joined = " ".join(
            generate_tokenized(src_slice_tokenized, constr_sliced_tokenized)
        )
        return f'{tokenized_joined}{nl if src.endswith((nl, cr)) else ""}'

    @dispatch(list)
    def tokenize(self, src: list[str]) -> list[str]:
        retval = []
        for sent in src:
            retval += (self.tokenize(sent),)
        return retval

    @dispatch(list, list, list)
    def tokenize(
        self,
        src: list[str],
        tgt: list[str],
        constraints: list[dict[tuple[tuple[int, int], tuple[int, int]], str]],
    ) -> tuple[list[str], list[str]]:
        src_constraints = self.generate_tuples(constraints, 0)
        # if we should use it in tgt?
        tgt_constraints = self.generate_tuples(constraints, 1)
        return (
            self.tokenize(src, src_constraints),
            self.tokenize(tgt, tgt_constraints),
        )

    @dispatch(list, list)
    def tokenize(
        self,
        src: list[str],
        constraints: list[dict[tuple[int, int], str]],
    ) -> list[str]:
        """Tokenizes the input with constraints

        Args:
            src (list): list of raw source sentences
            constraints (list): list of constraints of shape (range -> constraint)

        Returns:
            output (list): tokenized sentences
        """
        if len(src) != len(constraints):
            # since we are iterating over constarints, we need to add empty constraints,
            # so their count will correspond to that of input sentences
            while len(src) != len(constraints):
                constraints += ({},)

        retval = []
        for sent, constr in zip(src, constraints):
            retval += (self.tokenize(sent, constr),)
        return retval

    @dispatch(str, str)
    def detokenize(self, src_path: str, tgt_path: str) -> None:
        """Detokenizes the source file

        Args:
            src_path (str): path to source file
            tgt_path (str): path to target file
        """
        with open(src_path, "r") as src, open(tgt_path, "r") as tgt:
            self.detokenize(src, tgt)

    @dispatch(TextIOWrapper, TextIOWrapper)
    def detokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
        src_iter, src_iter_copy = tee(iter(src))
        n_lines = count_lines(src_iter_copy)
        for src_sent in wrap_tqdm(
            to_be_wrapped=src_iter, desc=f"Detokenizing {src.name}", n_lines=n_lines
        ):
            tgt.write(self.detokenize(src_sent))

    @dispatch(str)
    def detokenize(self, src: str) -> str:
        def extract_subword_n_factors(token: str) -> tuple[str, list[str]]:
            try:
                subword, factors = token.split("|", 1)
            except ValueError:
                subword, factors = token, "|"
            return subword, factors.split("|")

        def find_any(factors: list, *factors2find: list[str]) -> bool:
            return any(factor in factors2find for factor in factors)

        def assign_join(token: Token, factors: list[str]) -> None:
            token.join_left = True if "gl+" in factors else False
            token.join_right = True if "gr+" in factors else False

        def process_tokens(tokens: list[str]) -> list[Token]:
            tokens = iter(tokens)
            for token in tokens:
                subword, factors = extract_subword_n_factors(token)
                new_token = Token()

                # byte sequence
                if re.search(unk, token):
                    byte_sequence_factors = extract_subword_n_factors(token)[1]
                    byte_sequence = "".join(
                        re.search(r"(?<=<)\d(?=>)", next_token).group(0)
                        for next_token in takewhile(lambda t: t != "<#>", tokens)
                    )
                    try:
                        new_token.surface = chr(int(byte_sequence))
                    # invalid byte sequence (crucial for neural generated translations)
                    except (OverflowError, ValueError, TypeError):
                        new_token.surface = ""
                    else:
                        new_token.type = TokenType.WORD
                        new_token.spacer = True
                        new_token.casing = Casing.NONE
                        # make it empty space if byte sequence is newline/carriage return
                        new_token.surface = (
                            " " if new_token.surface in [nl, cr] else new_token.surface
                        )
                    assign_join(new_token, byte_sequence_factors)

                elif find_any(factors, "wbn", "wb"):
                    # assign casing and surface
                    if find_any(factors, "scu", "ca"):
                        new_token.casing = Casing.UPPERCASE
                        new_token.surface = subword
                    elif "ci" in factors:
                        new_token.casing = Casing.CAPITALIZED
                        new_token.surface = subword.lower().capitalize()
                    elif find_any(factors, "scl", "cn"):
                        new_token.casing = Casing.LOWERCASE
                        new_token.surface = subword.lower()
                    else:
                        new_token.casing = Casing.NONE
                        new_token.surface = subword
                    # word beginning/trailing subword
                    tokens_copy = copy.copy(tokens)
                    if "wbn" in factors:
                        new_token.type = TokenType.TRAILING_SUBWORD
                        new_token.join_left = True
                        new_token.spacer = False
                    else:
                        new_token.join_left = False
                        new_token.spacer = True
                        try:
                            next_token = next(tokens_copy)
                        except StopIteration:
                            new_token.type = TokenType.WORD
                        else:
                            if "wbn" in extract_subword_n_factors(next_token)[1]:
                                new_token.type = TokenType.LEADING_SUBWORD
                            else:
                                new_token.type = TokenType.WORD

                # punctuation, emoji
                else:
                    new_token.surface = subword
                    new_token.type = TokenType.WORD
                    new_token.spacer = True
                    new_token.casing = Casing.NONE
                    assign_join(new_token, factors)

                yield new_token

        tokens = src.split()
        tokens = list(process_tokens(tokens))
        detokenized_joined = "".join(self.tokenizer.detokenize(tokens))
        return f'{detokenized_joined}{nl if src.endswith((nl, cr)) else ""}'

    @dispatch(list)
    def detokenize(self, src: list[str]) -> list[str]:
        retval = []
        for sent in src:
            retval += (self.detokenize(sent),)
        return retval

    def generate_constraints(
        self,
        src_path: str,
        tgt_path: str,
        vocab_path: str,
        constraints_path: str,
        return_list: bool = True,
        only_src_spans: bool = False,
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
                if src_word != tgt_word:
                    vocab[src_word].add(tgt_word)
                if lemmatize:
                    for src_word_lemma in src_lemmatizer.lemmatize(src_word):
                        src_word_lemma = src_word_lemma.decode()
                        vocab[src_word_lemma] = vocab[src_word_lemma].union(
                            tgt_word_lemma.decode()
                            for tgt_word_lemma in tgt_lemmatizer.lemmatize(tgt_word)
                            if tgt_word_lemma.decode() != src_word_lemma
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
            tokenizer = Tokenizer(mode="aggressive")
            for src_sent, tgt_sent in wrap_tqdm(
                to_be_wrapped=zip(src_iter, tgt),
                desc=f"Generating constraints for {src.name}",
                n_lines=n_lines,
            ):
                # sentence skip
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
                                rv[
                                    src_span if only_src_spans else (src_span, tgt_span)
                                ] = constraint
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

    def spm_train(
        self,
        files: list[str],
        vocab_size: int,
        character_coverage: float,
        train_extremely_large_corpus: bool,
    ):
        """Trains an SP model

        Args:
            files (list): list of files to use for the training
            vocab_size (int): size of the resulting vocabulary
            character_coverage (float): coverage of words
        """
        learner = SentencePieceLearner(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            train_extremely_large_corpus=train_extremely_large_corpus,
            byte_fallback=True,
        )
        for file in files:
            if not os.path.exists(file):
                continue
            print(f"Ingesting file {file} ...", file=sys.stderr)
            learner.ingest_file(file)
        print(
            f"Training started. SP model will be saved to {self.model_path}.",
            file=sys.stderr,
        )
        learner.learn(self.model_path)
        # initialize the tokenizer from trained model
        self.tokenizer = Tokenizer(
            mode="aggressive",
            sp_model_path=self.model_path,
            case_feature=self.case_feature,
        )

    @staticmethod
    def generate_tuples(
        constraints: list[dict[tuple[tuple[int, int], tuple[int, int]], str]],
        idx: int,
    ) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
        for constraint in constraints:
            if type(constraint) != dict:
                constraint = eval(constraint)
            yield {c[0][idx]: c[1] for c in constraint.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--tokenize", action="store_true", help="Tokenize a file")
    action.add_argument("--detokenize", action="store_true", help="Detokenize a file")
    action.add_argument(
        "--generate", action="store_true", help="Generate a constraints file"
    )
    action.add_argument("--spm_train", action="store_true", help="Train an SP model")
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
        default="",
        type=str,
        help="Path to the constraints file",
    )
    parser.add_argument("--multispan_index", default=None, type=int)
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
        help="Path to the SP model",
    )
    parser.add_argument("--add_in", action="store_true", default=False)
    parser.add_argument("--add_constr", action="store_true", default=False)
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
    parser.add_argument("--only_src_spans", action="store_true", default=False)

    parser.add_argument("--vocab_size", default=32000, type=int, help="Vocabulary size")
    parser.add_argument(
        "--character_coverage",
        default=0.98,
        type=float,
        help="Character coverage to determine the minimum symbols",
    )
    parser.add_argument(
        "--train_extremely_large_corpus",
        action="store_true",
        default=False,
        help="Increase bit depth for unigram tokenization",
    )
    parser.add_argument("--train_sets", nargs="*", help="Files to be ingested")

    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
        help="Turn off the tqdm progress bar",
    )
    return parser.parse_args()


# cli
if __name__ == "__main__":
    args = parse_args()
    tokenizer = FactoredTokenizer(
        model_path=args.model_path,
        add_in=args.add_in,
        add_constr=args.add_constr,
        case_feature=args.case_feature,
    )

    if args.silent:
        silent = True

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
            only_src_spans=args.only_src_spans,
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
    elif args.spm_train:
        tokenizer.spm_train(
            files=args.train_sets,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            train_extremely_large_corpus=args.train_extremely_large_corpus,
        )
    else:
        with (
            open(args.src_path, "r")
            if args.src_path is not sys.stdin
            else sys.stdin as src,
            open(args.tgt_path, "w")
            if args.tgt_path is not sys.stdout
            else sys.stdout as tgt,
        ):
            if args.tokenize:
                if args.constraints_path:
                    with open(args.constraints_path, "r") as constraints:
                        if args.multispan_index != None:
                            tokenizer.tokenize(
                                src, tgt, constraints, args.multispan_index
                            )
                        else:
                            tokenizer.tokenize(src, tgt, constraints)
                else:
                    tokenizer.tokenize(src, tgt)
            else:
                tokenizer.detokenize(src, tgt)
