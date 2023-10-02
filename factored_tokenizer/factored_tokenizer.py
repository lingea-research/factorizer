#!/usr/bin/env python3

import argparse
from typing import Iterable, Optional, Iterator, Union
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

try:
    from .lemmatizer import Lemmatizer
except ImportError:
    from lemmatizer.lemmatizer import Lemmatizer


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
            self.tokenizer = Tokenizer(mode="aggressive", case_feature=case_feature)
        self.add_in = add_in
        self.reserved_symbols = reserved_symbols

    @dispatch(str, str, str)
    def tokenize(self, src_path: str, tgt_path: str, constraints_path: str) -> None:
        with (
            open(src_path, "r") as src,
            open(tgt_path, "w") as tgt,
            open(constraints_path, "r") as constraints
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
                    ] and not token.join_left:
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
    def tokenize(
        self,
        src: str,
        constraints: dict[
            Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], str
        ],
    ) -> str:
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
                # if src == constraint -> assign |t2 factor
                if s == c:
                    s = " ".join(assign_factors(s, "t2"))
                # if there is some constraint -> join both, assign the |t1 and |t2 factors
                # to the source and constraint respectively
                elif c:
                    s = " ".join(
                        [
                            *assign_factors(s, "t1"),
                            *assign_factors(c, "t2"),
                        ]
                    )
                # no constraint -> assign |t0 factor
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
            if not silent:
                print(f"Ingesting file {file} ...", file=sys.stderr)
            learner.ingest_file(file)
        if not silent:
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
        src_iter, src_iter_copy = tee(iter(constraints))
        n_lines = count_lines(src_iter_copy)
        for constraint in wrap_tqdm(
            to_be_wrapped=src_iter,
            desc=f"Separating constraint ranges with index {idx}",
            n_lines=n_lines,
        ):
            if type(constraint) != dict:
                constraint = eval(constraint)
            yield {c[0][idx]: c[1] for c in constraint.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--tokenize", action="store_true", help="Tokenize a file")
    action.add_argument("--detokenize", action="store_true", help="Detokenize a file")
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

    if args.spm_train:
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
