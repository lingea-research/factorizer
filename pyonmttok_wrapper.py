#!/usr/bin/env python3

import argparse
from typing import Iterable, Union, TextIO
import pyonmttok
from itertools import takewhile, islice
import re
import os
from io import TextIOWrapper
import sys
import codecs
import random
from multipledispatch import dispatch


random.seed(1234)
nl = '\n'
cr = '\r'
unk = '{unk,gl,gr}'

class PyonmttokWrapper:

	def __init__(self, model: str=None, add_in: bool=False, case_feature: bool=True,
	             reserved_symbols: list=['#', ':', '_', '\\', '|', '▁']):
		self.tokenizer = pyonmttok.Tokenizer(mode='aggressive', sp_model_path=model, case_feature=case_feature)
		self.add_in = add_in
		self.reserved_symbols = reserved_symbols

	@dispatch(str, str, str, float, float)
	def tokenize(self, src_path: str, tgt_path: str, constr_path: str='',
	             wskip: float=0.0, sskip: float=0.0) -> None:
		if constr_path:
			self.tokenize(open(src_path, 'r'), open(tgt_path, 'w'), open(constr_path), wskip, sskip)
		else:
			self.tokenize(open(src_path, 'r'), open(tgt_path, 'w'))

	@dispatch(TextIOWrapper, TextIOWrapper, TextIOWrapper, float, float)
	def tokenize(self, src: TextIOWrapper, tgt: TextIOWrapper, constr: TextIOWrapper=None,
	                  wskip: float=0.0, sskip: float=0.0) -> None:
		"""Tokenize file with constraints
		"""
		for s, c in zip(src, constr):
			tgt.write(self.tokenize(s, [eval(c)], wskip, sskip))
		src.close()
		tgt.close()
		constr.close()

	@dispatch(TextIOWrapper, TextIOWrapper)
	def tokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
		"""Tokenize file with no constraints
		"""
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
			src (list|TextIO): either a source file path or a list of input sentences

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
			bits, cnt = f'{int(byte, 16):0>8b}'[:4], 0
			while cnt < 4 and bits[cnt] != '0':
				cnt += 1
			return cnt if cnt > 0 else 1

		def byte_sequence2dec_sequence(byte_sequence: list[str]) -> str:
			"""Translates hexadecimal byte sequence to a decimal sequence

			Args:
				byte_sequence (list): byte sequence

			Returns:
				decimal sequence (str): decimal representation of the byte sequence
			"""

			return f'{" ".join([f"<{c}>" for c in str(ord(bytearray.fromhex("".join(byte_sequence)).decode("utf8")))])} <#>'

		def search_byte_pattern(txt: str) -> re.Match:
			"""Searches for <0xDD> pattern in `txt`

			Args:
				txt (str): text

			Returns:
				bytes (re.Match): found byte sequence
			"""

			return re.search(r'(?<=\<0[x])[\da-f]{2}(?=\>)', txt, flags=re.IGNORECASE)

		def get_join_factors(token: pyonmttok.Token) -> str:
			"""Gets string representation of join factors

			Args:
				token (Token): token object

			Returns:
				factors (str): string representation of join factors
			"""

			join_factors = '|gl+' if token.join_left else '|gl-'  # if join left
			join_factors += '|gr+' if token.join_right else '|gr-'  # if join right
			return join_factors

		def process_tokens(tokens: list[pyonmttok.Token]) -> Iterable[pyonmttok.Token]:
			tokens = iter(tokens)
			for token in tokens:
				byte = None

				# reserved symbols
				if token.surface in self.reserved_symbols:
					byte = token.surface.encode()
					hex = codecs.encode(byte, 'hex').decode()
					token.surface = f'<0x{hex}>'

				# alpha
				if token.surface.isalpha():
					if len(token.surface) > 1:  # word/subword
						token.features += '|ca' if token.casing == pyonmttok.Casing.UPPERCASE \
							else '|ci' if token.casing == pyonmttok.Casing.CAPITALIZED \
							else '|cn',
					else:  # single character
						token.features += '|scu' if token.casing in [pyonmttok.Casing.CAPITALIZED, pyonmttok.Casing.UPPERCASE] \
							else '|scl',
					token.features += '|wb' if token.type in [pyonmttok.TokenType.LEADING_SUBWORD, pyonmttok.TokenType.WORD] \
						else '|wbn',
					token.features += '|in' if self.add_in else '',

				# numeric
				elif token.surface.isnumeric():
					token.features +='|wb' if token.type in [pyonmttok.TokenType.LEADING_SUBWORD, pyonmttok.TokenType.WORD] \
						else '|wbn',
					token.features += '|in' if self.add_in else '',

				# unicode (find first byte in byte sequence)
				elif byte := search_byte_pattern(token.surface):
					token_sequence_length = parse_bits(byte.group())  # number of tokens to be skipped
					token.features = [unk, get_join_factors(token)]
					token_sequence = [token, *islice(tokens, 0, token_sequence_length-1)]
					token.surface = byte_sequence2dec_sequence(search_byte_pattern(token.surface).group() for token in token_sequence)

				# other
				else:
					token.features += get_join_factors(token),
					token.features += '|in' if self.add_in else '',

				new_surface = token.surface.upper() if token.surface.upper().lower() == token.surface else token.surface
				token.surface = f'{new_surface}{"".join(token.features)}' if not byte else f'{"".join(token.features)} {token.surface}'
				yield token

		retval = []
		for sent in src:
			tokens = self.tokenizer.tokenize(sent, as_token_objects=True)
			tokens = process_tokens(tokens)

			retval += f'{" ".join([token.surface for token in tokens if token.surface and not search_byte_pattern(token.surface)])}' \
			      		f'{nl if sent.endswith((nl, cr)) else ""}',
		return retval

	@dispatch(list, list, float, float)
	def tokenize(self, src: list[str], constraints: list[dict[tuple[int, int], list[str]]],
		           wskip: float=0.0, sskip: float=0.0) -> list[str]:
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
			"""Generates slices for sentences. It is assumed that the only delimiter between words is whitespace

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
					between = sent[prev_start_idx:key[0]]
					if between.startswith(' '):
						yield ['', '']
					if between.strip():
						yield [between.strip(), '']
					if between.endswith(' ') and len(between) > 1:
						yield ['', '']
					# yield constraint range
					yield [sent[key[0]:key[1]], val]
					prev_start_idx = key[1]

				# add last slice, even if its empty
				if sent[prev_start_idx:].startswith(' '):
					prev_start_idx += 1
					yield ['', '']
				yield [sent[prev_start_idx:].strip(), '']

			for sent, constr in zip(src, constraints):
				yield list(_generate_slices())

		def generate_tokenized() -> Iterable[str]:
			"""
			"""

			add_space = True
			byte_seq_pattern = r'^<[\d\#]>$'
			skip_sent = sskip > random.random()
			for s, c in zip(slice_tokenized, constr_tokenized):
				if not s:
					add_space = True
					continue
				# if constraint -> join both
				if c and not skip_sent and wskip < random.random():
					s = ' '.join([*[f'{s_sw}|t1' if not re.search(byte_seq_pattern, s_sw) else s_sw for s_sw in s.split()],
												*[f'{c_sw}|t2' if not re.search(byte_seq_pattern, c_sw) else c_sw for c_sw in c.split()]])
				else:
					s = ' '.join([f'{s_sw}|t0' if not re.search(byte_seq_pattern, s_sw) else s_sw for s_sw in s.split()])

				first, *others = s.split(' ', 1)
				if add_space:
					first = first.replace('|gl+', '|gl-')
					first = first.replace('|wbn', '|wb')
					add_space = False
				else:
					first = first.replace('|gl-', '|gl+')
					first = first.replace('|wb', '|wbn')
				yield ' '.join([first, *others])

		if len(src) != len(constraints):
			# since we are iterating over constarints, we need to add empty constraints,
			# so their count will correspond to that of input sentences
			while len(src) != len(constraints):
				constraints += {},

		retval = []
		src_slices = generate_slices()
		for slice in src_slices:
			sliced_src, sliced_constr = (list(s) for s in list(zip(*slice)))  # transpose
			slice_tokenized, constr_tokenized = list(self.tokenize(sliced_src)), \
																					list(self.tokenize(sliced_constr))
			retval += f'{" ".join(generate_tokenized())}{nl if src[0].endswith((nl, cr)) else ""}',
		return retval

	@dispatch(str, str)
	def detokenize(self, src_path: str, tgt_path: str) -> None:
		"""Detokenizes the source file

		Args:
			src_path (str): path to source file
			tgt_path (str): path to target file
		"""
		self.detokenize(open(src_path, 'r'), open(tgt_path, 'w'))

	@dispatch(TextIOWrapper, TextIOWrapper)
	def detokenize(self, src: TextIOWrapper, tgt: TextIOWrapper) -> None:
		for s in src:
			tgt.write(self.detokenize(s))
		src.close()
		tgt.close()

	@dispatch(str)
	def detokenize(self, src: str) -> Iterable[str]:
		return self.detokenize([src])[0]

	@dispatch(list)
	def detokenize(self, src: list[str]) -> list[str]:
		"""Detokenizes sentence(s)

		Args:
			src (list): list of input sentences

		Returns:
			output (list): detokenized sentences
		"""
		def extract_subword_n_factors(token: str):
			try:
				subword, factors = token.split('|', 1)
			except ValueError:
				subword, factors = token, '|'
			return subword, factors.split('|')

		def find_any(factors: list, *factors2find: list):
			return any(factor in factors2find for factor in factors)

		def assign_join(token: pyonmttok.Token, factors: list[str]):
			token.join_left = True if 'gl+' in factors else False
			token.join_right = True if 'gr+' in factors else False

		def process_tokens(tokens: list[str]) -> list[pyonmttok.Token]:
			tokens = iter(tokens)
			for token in tokens:
				new_token = pyonmttok.Token()
				# byte sequence
				if re.search(unk, token):
					byte_sequence_factors = extract_subword_n_factors(token)[1]
					byte_sequence = ''.join(re.search(r'(?<=<)\d(?=>)', next_token).group(0)
			                            for next_token in takewhile(lambda t: t != '<#>', tokens))
					try:
						new_token.surface, new_token.type, new_token.spacer, new_token.casing = \
						  chr(int(byte_sequence)), pyonmttok.TokenType.WORD, True, pyonmttok.Casing.NONE
						# make it empty space if byte sequence is newline/carriage return
						new_token.surface = ' ' if new_token.surface in [nl, cr] else new_token.surface
					# invalid byte sequence
					except OverflowError:
						new_token.surface = ''
					assign_join(new_token, byte_sequence_factors)
					yield new_token
					continue

				subword, factors = extract_subword_n_factors(token)
				if find_any(factors, 'wbn', 'wb'):
					# assign casing and surface
					new_token.casing, new_token.surface = (pyonmttok.Casing.UPPERCASE, subword) if find_any(factors, 'scu', 'ca') \
					  else (pyonmttok.Casing.CAPITALIZED, subword.lower().capitalize()) if 'ci' in factors \
					  else (pyonmttok.Casing.LOWERCASE, subword.lower()) if find_any(factors, 'scl', 'cn') \
					  else (pyonmttok.Casing.NONE, subword)
					# assign type, join_left and spacer
					new_token.type, new_token.join_left, new_token.spacer = (pyonmttok.TokenType.TRAILING_SUBWORD, True, False) if 'wbn' in factors \
						else (pyonmttok.TokenType.WORD, False, True)

				# punctuation, emoji
				else:
					new_token.surface, new_token.type, new_token.spacer, new_token.casing = subword, pyonmttok.TokenType.WORD, True, pyonmttok.Casing.NONE
					assign_join(new_token, factors)

				yield new_token

		retval = []
		for sent in src:
			tokens = sent.split()
			tokens = list(process_tokens(tokens))
			# add newline if there is one in the source sentence
			retval += f'{self.tokenizer.detokenize(tokens)}{nl if sent.endswith((nl, cr)) else ""}',
		return retval


def parse_args():
	parser = argparse.ArgumentParser()
	tokenize = parser.add_mutually_exclusive_group(required=True)
	tokenize.add_argument('--tokenize', action='store_true')
	tokenize.add_argument('--detokenize', action='store_true')
	parser.add_argument('-s', '--src', default=sys.stdin, type=str, help='Either a path to source file or string to be processed')
	parser.add_argument('-t', '--tgt', default=sys.stdout, type=str, help='Path to target file')
	parser.add_argument('-c', '--constraints', default='', type=str, help='Path to constraints file')
	parser.add_argument('--wskip', default=0.0, type=float, help='Word skip probability, for training only')
	parser.add_argument('--sskip', default=0.0, type=float, help='Sentence skip probability, for training only')
	parser.add_argument('-m', '--model', default=None, type=str, help='Path to SP model')
	parser.add_argument('--add_in', action='store_true', default=False)
	parser.add_argument('--no_case_feature', action='store_false', dest='case_feature', default=True)
	return parser.parse_args()

# cli
if __name__ == '__main__':
	args = parse_args()
	tokenizer = PyonmttokWrapper(model=args.model, add_in=args.add_in, case_feature=args.case_feature)
	tokenizer.tokenize(args.src, args.tgt, args.constraints, args.wskip, args.sskip) if args.tokenize \
		else tokenizer.detokenize_cli(args.src, args.tgt) if args.detokenize \
		else None
