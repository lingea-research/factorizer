#!/usr/bin/env python3

import argparse
import string
from typing import Iterable, Union
from pyonmttok import Tokenizer, TokenType, Casing, SentencePieceLearner
from itertools import takewhile, islice
import re
import sys
import copy
from tqdm import tqdm
import os
import codecs

from factorizer.types import Token, Factors
from factorizer.patterns import byteseq_byte_pattern, continuous_scripts_pattern


nl = "\n"


class Factorizer:
    def __init__(
        self,
        sp_model_path: str = None,
        # will be appended to ALL tokens except for the special ones (unk, ..)
        factors_to_add_soft: list[str] = [],
        # will be appended to ALL tokens
        factors_to_add_hard: list[str] = [],
        case_feature: bool = True,
        reserved_symbols: list[str] = ["#", ":", "_", "\\", "|", "▁"],
        preserve_placeholders: bool = True,
        segment_numbers: bool = False,
        case_insensetive: bool = False,
    ):
        self.factors_to_add_soft = [
            f"|{f}" if not f.startswith("|") else f for f in factors_to_add_soft
        ]
        self.factors_to_add_hard = [
            f"|{f}" if not f.startswith("|") else f for f in factors_to_add_hard
        ]
        self.sp_model_path = sp_model_path
        self.reserved_symbols = reserved_symbols
        self.case_insensetive = case_insensetive
        self.onmt_args = {
            "mode": "aggressive",
            "case_feature": case_feature,
            "segment_numbers": segment_numbers,
            "preserve_placeholders": preserve_placeholders,
            "sp_model_path": sp_model_path,
        }
        self.tokenizer = None
        if sp_model_path and os.path.exists(sp_model_path):
            self.tokenizer = Tokenizer(**self.onmt_args)

    def tokenize(
        self,
        src: str,
        constraints: Union[str, dict[tuple[int, int], str]] = {},
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("ONMT Tokenizer was not initialized")
        if isinstance(constraints,  str):
            constraints = eval(constraints)
        return (
            self.__tokenize_constraints(src, constraints)
            if constraints
            else self.__tokenize(src)
        )

    def tokenize_batch(
        self,
        src: list[str],
        constraints: list[Union[str, dict[tuple[int, int], str]]] = [],
    ) -> list[str]:
        """Tokenizes the input with constraints

        Args:
            src (list): list of raw source sentences
            constraints (list): list of constraints of shape (range: constraint)

        Returns:
            output (list): tokenized sentences
        """
        if constraints and isinstance(constraints, str):
            # since we are iterating over constarints,
            # we need to add empty constraints,
            # so their count will correspond to that of input sentences
            constraints = [eval(c) for c in constraints]
            while len(src) != len(constraints):
                constraints += ({},)
        return list(
            map(self.__tokenize_constraints, src, constraints)
            if constraints
            else map(self.__tokenize, src)
        )

    def __tokenize(self, src: str) -> str:
        def search_byte_pattern(txt: str) -> str | None:
            """Searches for <0xDD> pattern in `txt`

            Args:
                txt (str): text

            Returns:
                bytes (re.Match): found byte sequence
            """

            search_result = re.search(
                r"(?<=\<0x)[\da-f]{2}(?=\>)",
                txt,
                flags=re.IGNORECASE,
            )
            return search_result.group() if search_result else None


        def process_tokens(tokens: list[Token]) -> Iterable[Token]:
            def parse_utf8_bits(byte: str) -> int:
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
                    decimal sequence (str): dec representation of
                      the byte sequence
                """
                bytes_str = str(
                    ord(bytearray.fromhex(
                        "".join(byte_sequence)
                    ).decode("utf8"))
                )
                return (
                    f"{' '.join([f'<{c}>' for c in bytes_str])} "
                    f"{Token.unk_sequence_end}"
                )

            def get_join_factors(token: Token) -> list[str]:
                """Gets string representation of join factors

                Args:
                    token (Token): token object

                Returns:
                    factors (str): string representation of join factors
                """
                # check next (non-byte) token, if word+join_left -> add gl+
                tokens_copy = copy.copy(tokens)
                try:
                    next_token = next(tokens_copy)
                    while re.search(
                        r"(?<=\<0x)[\da-f]{2}(?=\>)",
                        next_token.surface, flags=re.IGNORECASE
                    ):
                        next_token = next(tokens_copy)
                except StopIteration:
                    next_token = None

                join_factors = [
                    Factors.glue_left if token.join_left
                    else Factors.glue_left_not
                ]
                join_factors += (Factors.glue_right if token.join_right
                                 or (next_token
                                     and next_token.join_left
                                     and next_token.type in [
                                         TokenType.LEADING_SUBWORD,
                                         TokenType.WORD
                                     ])
                                 else Factors.glue_right_not,)
                return join_factors

            def parse_byte_sequence(byte: str) -> str:
                token_sequence_length = parse_utf8_bits(byte)
                token.features = [Token.unk_lemma, *get_join_factors(token)]
                token_sequence = [
                    token, *islice(tokens, 0, token_sequence_length - 1)
                ]
                return byte_sequence2dec_sequence(
                    search_byte_pattern(token.surface)
                    for token in token_sequence
                )

            tokens = iter(tokens)
            join_left_after_numeric = False
            for token in tokens:
                byte = None
                continuous_script = continuous_scripts_pattern.search(
                    token.surface
                )
                if token.surface in self.reserved_symbols:
                    byte = token.surface.encode()
                    hex = codecs.encode(byte, "hex").decode()
                    token.surface = f"<0x{hex}>"
                    token.surface = parse_byte_sequence(hex)
                # word
                elif token.surface.isalpha():
                    # word/subword
                    if len(token.surface) > 1:
                        match (token.casing):
                            case Casing.UPPERCASE:
                                token.features += (Factors.case_upper,)
                            case Casing.CAPITALIZED:
                                token.features += (Factors.case_capitalized,)
                            case Casing.LOWERCASE:
                                token.features += (Factors.case_lower,)
                    # single character
                    elif not continuous_script:
                        match (token.casing):
                            case Casing.UPPERCASE | Casing.CAPITALIZED:
                                token.features += (Factors.signle_upper,)
                            case Casing.LOWERCASE:
                                token.features += (Factors.single_lower,)
                    # beginning of word
                    if (
                        token.type in [
                            TokenType.LEADING_SUBWORD,
                            TokenType.WORD,
                        ]
                        and not join_left_after_numeric
                    ):
                        token.features += (
                            Factors.word_beg
                            if not continuous_script
                            else Factors.continuous_script_beg,
                        )
                    else:
                        token.features += (
                            Factors.word_beg_not
                            if not continuous_script
                            else Factors.continuous_script_beg_not,
                        )
                        join_left_after_numeric = False
                # numeric
                elif token.surface.isnumeric():
                    if (
                        token.type
                        in [
                            TokenType.LEADING_SUBWORD,
                            TokenType.WORD,
                        ]
                        and not token.join_left
                    ):
                        token.features += (Factors.word_beg,)
                    else:
                        token.features += (Factors.word_beg_not,)
                    if token.join_right is True:
                        join_left_after_numeric = True
                # unicode (find the first byte in byte sequence)
                elif (byte := search_byte_pattern(token.surface)):
                    # number of tokens to be skipped
                    token.surface = parse_byte_sequence(byte)
                # other
                else:
                    token.features = get_join_factors(token)

                # add factors
                if token.features[0] != Token.unk_lemma:
                    token.features += self.factors_to_add_soft
                token.features += self.factors_to_add_hard

                new_surface = (
                    token.surface.upper()
                    if token.surface.upper().lower() == token.surface
                    else token.surface
                )

                token.surface = (
                    f'{new_surface}|{"|".join(token.features)}'
                    if not byte
                    else f'{"|".join(token.features)} {token.surface}'
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
        return (
            f"{tokenized_joined}"
            f"{nl if src.endswith(tuple(string.whitespace)) else ''}"
        )

    def __tokenize_constraints(
        self,
        src: str,
        constraints: dict[tuple[int, int], str],
    ) -> str:
        def generate_slices(
            sent: str, constr: dict[tuple[int, int], str]
        ) -> Iterable[tuple[str, str]]:
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
            src_slice_tokenized: list[str], constr_slice_tokenized: list[str]
        ) -> Iterable[str]:
            add_space = True

            def assign_constr(
                s: list[str],
                constr_factor: str,
            ) -> Iterable[str]:
                return (
                    f"{sw}|{constr_factor}" if not byteseq_byte_pattern.search(
                        sw,
                    ) else sw for sw in s.split()
                )

            for s, c in zip(src_slice_tokenized, constr_slice_tokenized):
                if not s:
                    add_space = True
                    continue
                # if there is some constraint, join both,
                # assign the |t1 and |t2 factors
                # to the source and constraint respectively
                if c:
                    s = " ".join([
                        *assign_constr(s, "t1"),
                        *assign_constr(c, "t2"),
                    ])
                # no constraint -> assign |t0 factor
                else:
                    s = " ".join(assign_constr(s, "t0"))

                first, *others = s.split(" ", 1)
                if add_space:
                    first = first.replace(
                        Factors.glue_left,
                        Factors.glue_left_not,
                    )
                    first = first.replace(
                        Factors.word_beg_not,
                        Factors.word_beg,
                    )
                    add_space = False
                else:
                    first = first.replace(
                        Factors.glue_left_not,
                        Factors.glue_left,
                    )
                    first = first.replace(
                        f"|{Factors.word_beg}",
                        f"|{Factors.word_beg_not}",
                    )
                yield " ".join([first, *others])

        slices = generate_slices(src, constraints)
        src_slice, constr_slice = (list(s) for s in list(zip(*slices)))
        src_slice_tokenized = self.tokenize_batch(src_slice)
        constr_sliced_tokenized = self.tokenize_batch(constr_slice)
        tokenized_joined = " ".join(
            generate_tokenized(src_slice_tokenized, constr_sliced_tokenized)
        )
        return (
            f"{tokenized_joined}"
            f"{nl if src.endswith(tuple(string.whitespace)) else ''}"
        )

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

        def process_tokens(tokens: list[str]) -> Iterable[Token]:
            tokens = iter(tokens)
            for token in tokens:
                subword, factors = extract_subword_n_factors(token)
                new_token = Token()

                # process byte sequence
                if re.search(Token.unk_lemma, token):
                    byte_sequence_factors = extract_subword_n_factors(token)[1]
                    byte_sequence = "".join(
                        re.search(r"(?<=<)\d(?=>)", next_token).group(0)
                        for next_token in
                        takewhile(lambda t: t != "<#>", tokens)
                    )
                    try:
                        new_token.surface = chr(int(byte_sequence))
                    # invalid byte sequence
                    # (crucial for neural generated translations)
                    except (OverflowError, ValueError, TypeError):
                        new_token.surface = ""
                    else:
                        new_token.type = TokenType.WORD
                        new_token.spacer = True
                        new_token.casing = Casing.NONE
                        new_token.surface = (
                            " " if new_token.surface in string.whitespace
                            else new_token.surface
                        )
                    assign_join(new_token, byte_sequence_factors)

                elif find_any(factors, Factors.word_beg, Factors.word_beg_not):
                    # assign casing and surface
                    if find_any(
                        factors,
                        Factors.signle_upper,
                        Factors.case_upper,
                    ):
                        new_token.casing = Casing.UPPERCASE
                        new_token.surface = subword
                    elif Factors.case_capitalized in factors:
                        new_token.casing = Casing.CAPITALIZED
                        new_token.surface = subword.lower().capitalize()
                    elif find_any(
                        factors,
                        Factors.single_lower,
                        Factors.case_lower,
                    ):
                        new_token.casing = Casing.LOWERCASE
                        new_token.surface = subword.lower()
                    else:
                        new_token.casing = Casing.NONE
                        new_token.surface = subword
                    # word beginning/trailing subword
                    tokens_copy = copy.copy(tokens)
                    if Factors.word_beg_not in factors:
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
                            if (
                                Factors.word_beg_not
                                in extract_subword_n_factors(next_token)[1]
                            ):
                                new_token.type = TokenType.LEADING_SUBWORD
                            else:
                                new_token.type = TokenType.WORD
                elif find_any(
                    factors,
                    Factors.continuous_script_beg,
                    Factors.continuous_script_beg_not,
                ):
                    new_token.casing = Casing.NONE
                    new_token.surface = subword
                    if Factors.continuous_script_beg in factors:
                        new_token.type = TokenType.LEADING_SUBWORD
                        new_token.spacer = True
                        new_token.join_left = False
                    else:
                        new_token.type = TokenType.TRAILING_SUBWORD
                        new_token.spacer = False
                        new_token.join_left = True
                # punctuation, emoji
                else:
                    new_token.surface = subword
                    new_token.type = TokenType.WORD
                    new_token.spacer = True
                    new_token.casing = Casing.NONE
                    assign_join(new_token, factors)

                yield new_token

        if self.tokenizer is None:
            raise RuntimeError("ONMT Tokenizer was not initialized.")

        tokens = src.split()
        tokens = list(process_tokens(tokens))
        detokenized_joined = "".join(self.tokenizer.detokenize(tokens))
        return (
            f"{detokenized_joined}"
            f"{nl if src.endswith(tuple(string.whitespace)) else ''}"
        )

    def detokenize_batch(self, src: list[str]) -> list[str]:
        return list(map(self.detokenize, src))

    def spm_train(
        self,
        files: list[str],
        vocab_size: int,
        character_coverage: float,
        train_extremely_large_corpus: bool,
        sp_model_path: str = None,
    ):
        """Trains an SP model on casefold files

        Args:
            files (list): list of files to use for the training
            vocab_size (int): size of the resulting vocabulary
            character_coverage (float): coverage of words
            train_extremely_large_corpus (bool): spm flag
            sp_model_path (str): path to sp model
        """

        if sp_model_path:
            self.onmt_args["sp_model_path"] = sp_model_path
        if (
            not self.onmt_args["sp_model_path"]
        ):
            raise RuntimeError("Model path was not provided")

        learner = SentencePieceLearner(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            train_extremely_large_corpus=train_extremely_large_corpus,
            byte_fallback=True,
        )
        for file_path in files:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as file_in:
                print(f"Ingesting file {file_path} ...", file=sys.stderr)
                for line in file_in:
                    learner.ingest(line.casefold())
        print(
            f"Training started. SP model will be "
            f"saved to {self.onmt_args['sp_model_path']}.",
            file=sys.stderr,
        )
        learner.learn(self.onmt_args["sp_model_path"])
        # initialize the tokenizer from trained model
        self.tokenizer = Tokenizer(**self.onmt_args)


def parse_args():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--tokenize",
        action="store_true",
        help="Tokenize a file",
    )
    action.add_argument(
        "--detokenize",
        action="store_true",
        help="Detokenize a file",
    )
    action.add_argument(
        "--spm_train",
        action="store_true",
        help="Train an SP model",
    )
    parser.add_argument(
        "-s",
        "--src",
        dest="src_path",
        default=sys.stdin,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--tgt",
        dest="tgt_path",
        default=sys.stdout,
        type=str,
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
        "--add_factors_soft",
        dest="factors_to_add_soft",
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--add_factors_hard",
        dest="factors_to_add_hard",
        nargs="*",
        default=[]
    )

    parser.add_argument(
        "--vocab_size",
        default=32000,
        type=int,
    )
    parser.add_argument(
        "--character_coverage",
        default=0.98,
        type=float,
        help="Amount of characters covered by the model",
    )
    parser.add_argument(
        "--train_extremely_large_corpus",
        action="store_true",
        default=False,
        help="Increase bit depth for unigram tokenization",
    )
    parser.add_argument(
        "--train_sets",
        dest="files",
        nargs="*",
        help="Files to be ingested",
    )

    parser.add_argument(
        "--no_case_feature",
        action="store_false",
        dest="case_feature",
        default=True,
    )
    parser.add_argument(
        "--reserved_symbols",
        nargs="*",
        default=["#", ":", "_", "\\", "|", "▁"],
        type=list,
    )
    parser.add_argument(
        "--segment_numbers",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_preserve_placeholders",
        dest="preserve_placeholders",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="sp_model_path",
        default=None,
        type=str,
        help="Path to the SP model",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def cli():
    def _tqdm(iterable, desc):
        return iterable if args.silent else tqdm(iterable=iterable, desc=desc)
    args = parse_args()
    tokenizer = Factorizer(
        sp_model_path=args.sp_model_path,
        factors_to_add_soft=args.factors_to_add_soft,
        factors_to_add_hard=args.factors_to_add_hard,
        case_feature=args.case_feature,
        reserved_symbols=args.reserved_symbols,
        segment_numbers=args.segment_numbers,
        preserve_placeholders=args.preserve_placeholders,
    )

    if args.spm_train:
        tokenizer.spm_train(
            files=args.files,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            train_extremely_large_corpus=args.train_extremely_large_corpus,
        )

    elif args.tokenize:
        with (
            open(args.src_path, "r")
            if args.src_path is not sys.stdin
            else sys.stdin as src,
            open(args.tgt_path, "w")
            if args.tgt_path is not sys.stdout
            else sys.stdout as tgt,
        ):
            if args.constraints_path:
                with open(args.constraints_path, "r") as constraints:
                    for s, c in _tqdm(
                        iterable=zip(src, constraints),
                        desc=f"Tokenizing {args.src_path}...",
                    ):
                        tgt.write(tokenizer.tokenize(s, c))
            else:
                for s in _tqdm(
                    iterable=src,
                    desc=f"Tokenizing {args.src_path}...",
                ):
                    tgt.write(tokenizer.tokenize(s))

    elif args.detokenize:
        with (
            open(args.src_path, "r")
            if args.src_path is not sys.stdin
            else sys.stdin as src,
            open(args.tgt_path, "w")
            if args.tgt_path is not sys.stdout
            else sys.stdout as tgt,
        ):
            for s in _tqdm(
                iterable=src,
                desc=f"Detokenizing {args.src_path}...",
            ):
                tgt.write(tokenizer.detokenize(s))


if __name__ == "__main__":
    cli()
