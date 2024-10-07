import regex as re

special_symbol_or_byte_pattern = re.compile(r"^<.+>$", re.UNICODE)
byte_seq_pattern = re.compile(r"\\x[a-fA-F\d]{2}", re.UNICODE)
byteseq_byte_pattern = re.compile(r"^<[\d\#]>$", re.UNICODE)
continuous_scripts_pattern = re.compile(
    r"([\p{IsHan}\p{IsBopo}\p{IsHira}\p{IsKatakana}]+)", re.UNICODE
)  # pattern for continuous scripts
