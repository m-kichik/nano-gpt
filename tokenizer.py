from typing import List


class CharTokenizer:
    def __init__(self, text: str):
        if text is None:
            raise ValueError("Empty text")
        self.vocab_size = None
        self.build(text)

    def build(self, text: str):
        alphabet = set(text)
        self.vocab_size = len(alphabet)
        self.symbol2int = {s: i for i, s in enumerate(alphabet)}
        self.int2symbol = {i: s for i, s in enumerate(alphabet)}

    def encode(self, text: str) -> List[int]:
        return [self.symbol2int[s] for s in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.int2symbol[i] for i in tokens])
