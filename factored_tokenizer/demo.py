from .factored_tokenizer import FactoredTokenizer as Tokenizer

def demo():
	tokenizer = Tokenizer(
		model='/home/large/data/models/marian/csen_factored_pyonmttok_fs.20221118/pyonmttok/csen-lowered.spm',
	    add_in=False,
        case_feature=True
    )
	input = ['Last Friday I😊 saw a spotted striped blue worm shake hands with a legless lizard.', '"That building is 5a tall."', "L 147, 17.5.2014, s. 79).“"]
	constr = [{(0,4): 'posledni', (5,11): 'Patek', (12,13): 'Ja', (13,14): '嘚', (37,41): 'modry'}, {(6,14): 'stavba', (0,1): '«', (26,27): '»'}, {}]
	print('Input:', input)
	print('Constraints:', constr)
	tokenized = tokenizer.tokenize(input, constr)
	print('Tokenized with constraints:', tokenized)
	tokenized = tokenizer.tokenize(input)
	print('Tokenized without constraints:', tokenized)
	detokenized = tokenizer.detokenize(tokenized)
	print('Detokenized:', detokenized)
	print('Is detokenized equal to the input?', input==detokenized)


if __name__ == '__main__':
	demo()