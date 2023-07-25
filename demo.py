from pyonmttok_wrapper import PyonmttokWrapper


def demo():
	tokenizer = PyonmttokWrapper(model='/home/large/data/models/marian/csen_factored_pyonmttok_fs.20221118/pyonmttok/csen-lowered.spm',
															 add_in=False, case_feature=True)
	print(['Last Friday I😊 saw a spotted striped blue worm shake hands with a legless lizard.', '"That building is tall."'])
	tokenized = tokenizer.tokenize(src=['Last Friday I😊 saw a spotted striped blue worm shake hands with a legless lizard.', '"That building is tall."'],
																 constraints=[{(0,4): 'posledni', (5,11): 'Patek', (12,13): 'Ja', (13,14): '嘚', (37,41): 'modry'}, {(6,14): 'stavba', (0,1): '«', (23,24): '»'}],
																 wskip=0.2, sskip=0.5)
	print(tokenized)


if __name__ == '__main__':
	demo()