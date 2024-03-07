import os
from typing import List


class CharTokenizer:
    def __init__(self, text_path: str):
        if text_path is None:
            raise ValueError("Empty path")
        with open(text_path, "r") as file:
            text = file.read()
        self.vocab_size = None
        self.build(text)

    def build(self, text: str):
        alphabet = list(set(text))
        alphabet.sort()
        self.vocab_size = len(alphabet)
        self.symbol2int = {s: i for i, s in enumerate(alphabet)}
        self.int2symbol = {i: s for i, s in enumerate(alphabet)}

    def encode(self, text: str) -> List[int]:
        return [self.symbol2int[s] for s in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.int2symbol[i] for i in tokens])
