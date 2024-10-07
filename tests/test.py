import sys
import os
from factorizer import Factorizer


def assert_output(output, reference):
    assert output == reference, f"\nout: {output}\nref: {reference}"


def test_module():
    tokenizer = Factorizer(
        sp_model_path=os.path.join(
            os.path.dirname(__file__),
            "models/test.spm"
        ),
        case_feature=True,
    )
    input = [
        "Last Friday I😊 saw a spotted striped blue worm shake hands with a legless lizard.",
        "'That building is 5a tall.'",
        "L 147, 17.5.2014, s. 79).“"
    ]
    constraints = [
        {(0,4): "posledni", (5,11): "Patek", (12,13): "Ja", (13,14): "嘚"},
        {(6,14): "stavba", (0,1): "«", (26,27): "»"},
        {}
    ]
    tokenized_batch = tokenizer.tokenize_batch(input)
    tokenized_constraints_batch = tokenizer.tokenize_batch(input, constraints)
    tokenized = tokenizer.tokenize(input[0])
    tokenized_constraints = tokenizer.tokenize(input[0], constraints[0])
    detokenized_batch = tokenizer.detokenize_batch(tokenized_batch)
    detokenized = tokenizer.detokenize(tokenized)

    tokenized_constraints_batch_ref = [
        "LAST|ci|wb|t1 PO|cn|wb|t2 SLED|cn|wbn|t2 NI|cn|wbn|t2 "
        "FRIDAY|ci|wb|t1 PAT|ci|wb|t2 EK|cn|wbn|t2 I|scu|wb|t1 "
        "JA|ci|wb|t2 {unk,gl,gr}|gl+|gr-|t1 <1> <2> <8> <5> <2> <2> <#> "
        "{unk,gl,gr}|gl-|gr-|t2 <2> <2> <0> <4> <2> <#> SAW|cn|wb|t0 "
        "A|scl|wb|t0 SPOTTED|cn|wb|t0 STRIP|cn|wb|t0 ED|cn|wbn|t0 "
        "BLUE|cn|wb|t0 WORM|cn|wb|t0 SHAKE|cn|wb|t0 HANDS|cn|wb|t0 "
        "WITH|cn|wb|t0 A|scl|wb|t0 LEG|cn|wb|t0 LESS|cn|wbn|t0 "
        "LIZ|cn|wb|t0 ARD|cn|wbn|t0 .|gl+|gr-|t0",

        "'|gl-|gr-|t1 {unk,gl,gr}|gl-|gr-|t2 <1> <7> <1> <#> THAT|ci|wbn|t0 "
        "BUILDING|cn|wb|t1 STAVBA|cn|wb|t2 IS|cn|wb|t0 5|wb|t0 A|scl|wbn|t0 "
        "TALL|cn|wb|t0 .|gl+|gr-|t0 "
        "'|gl+|gr-|t1 {unk,gl,gr}|gl-|gr-|t2 <1> <8> <7> <#>",

        "L|scu|wb|t0 147|wb|t0 ,|gl+|gr-|t0 17|wb|t0 .|gl+|gr+|t0 5|wb|t0 "
        ".|gl+|gr+|t0 2014|wb|t0 ,|gl+|gr-|t0 S|scl|wb|t0 .|gl+|gr-|t0 "
        "79|wb|t0 )|gl+|gr+|t0 .|gl+|gr+|t0 “|gl+|gr-|t0",
    ]
    assert_output(tokenized_constraints_batch, tokenized_constraints_batch_ref)

    tokenized_constraints_ref = (
        "LAST|ci|wb|t1 PO|cn|wb|t2 SLED|cn|wbn|t2 "
        "NI|cn|wbn|t2 FRIDAY|ci|wb|t1 PAT|ci|wb|t2 EK|cn|wbn|t2 I|scu|wb|t1 "
        "JA|ci|wb|t2 {unk,gl,gr}|gl+|gr-|t1 <1> <2> <8> <5> <2> <2> <#> "
        "{unk,gl,gr}|gl-|gr-|t2 <2> <2> <0> <4> <2> <#> SAW|cn|wb|t0 "
        "A|scl|wb|t0 SPOTTED|cn|wb|t0 STRIP|cn|wb|t0 ED|cn|wbn|t0 "
        "BLUE|cn|wb|t0 WORM|cn|wb|t0 SHAKE|cn|wb|t0 HANDS|cn|wb|t0 "
        "WITH|cn|wb|t0 A|scl|wb|t0 LEG|cn|wb|t0 LESS|cn|wbn|t0 "
        "LIZ|cn|wb|t0 ARD|cn|wbn|t0 .|gl+|gr-|t0"
    )
    assert_output(tokenized_constraints, tokenized_constraints_ref)

    assert_output(input[0], detokenized)

    assert_output(input, detokenized_batch)

def init():
    t = Factorizer()

tests = [
    ("module", test_module),
    ("empty_init", init),
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
