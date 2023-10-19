import sys
import os
from factored_tokenizer import FactoredTokenizer as Tokenizer

def test_module():
    tokenizer = Tokenizer(
        model_path=os.path.join(os.path.dirname(__file__), "models/csen-lowered.spm"),
        add_in=False,
        case_feature=True,
    )
    input = [
        "Last Friday I😊 saw a spotted striped blue worm shake hands with a legless lizard.",
        "'That building is 5a tall.'",
        "L 147, 17.5.2014, s. 79).“"
    ]
    # constraints = [
    #     {(0,4): "posledni", (5,11): "Patek", (12,13): "Ja", (13,14): "嘚"},
    #     {(6,14): "stavba", (0,1): "«", (26,27): "»"},
    #     {}
    # ]
    tokenized_batch = tokenizer.tokenize_batch(input)
    # tokenized_constraints_batch = tokenizer.tokenize_batch(input, constraints)
    tokenized = tokenizer.tokenize(input[0])
    # tokenized_constraints = tokenizer.tokenize(input[0], constraints[0])
    detokenized_batch = tokenizer.detokenize_batch(tokenized_batch)
    detokenized = tokenizer.detokenize(tokenized)
    assert input[0] == detokenized
    assert input == detokenized_batch


tests = [
    ("module", test_module),
]


if __name__ == "__main__":
    for required_test in [name for name, _ in tests]:
        for name, test in tests:
            if name == required_test:
                print("Running test {}".format(name), file=sys.stderr)
                test()
                break
        else:
            raise ValueError("Unknown test {}".format(name))
