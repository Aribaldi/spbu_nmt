from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
from torchtext.datasets import Multi30k, IWSLT2016
from torchtext.vocab import build_vocab_from_iterator
from .config import TGT_LANGUAGE, SRC_LANGUAGE, special_symbols, UNK_IDX



token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_lg')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_lg')

def yield_tokens(data_iter:Iterable, language:str) -> List[str]:
    language_index = {SRC_LANGUAGE:0, TGT_LANGUAGE:1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def get_vocab_transforms():
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) + \
                    IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
    return vocab_transform

vocab_transform = get_vocab_transforms()

if __name__ == '__main__':
    temp = token_transform['en']("To be or not to be; that's the question")
    print(temp)
    print(vocab_transform['en'](temp))
