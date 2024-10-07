from pyonmttok import Token as ONMTToken


class Token(ONMTToken):
    factor_separator = "|"
    unk_lemma = "{unk,gl,gr}"
    unk_digits = {f"<{i}>" for i in range(10)}
    unk_sequence_end = "<#>"

class Factors:
    word_beg = "wb"
    word_beg_not = "wbn"
    signle_upper = "scu"
    single_lower = "scl"
    glue_left = "gl+"
    glue_left_not = "gl-"
    glue_right = "gr+"
    glue_right_not = "gr-"
    case_upper = "ca"
    case_capitalized = "ci"
    case_lower = "cn"


    def __init__(self, new_factors: list[str]) -> None:
        self.add_factors = new_factors
