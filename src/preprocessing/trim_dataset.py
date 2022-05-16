from pathlib import Path
import os
from itertools import compress
from .config import TRIM_THRESH


def get_sentence_len(line):
    return len(line.split())


def trim_lang_pair(de_file_path, en_file_path):
    with open(de_file_path, mode='r') as df:
        with open(en_file_path, mode='r') as ef:
            de_lines = df.readlines()
            en_lines = ef.readlines()
            idx_mask = [0 for i in range(len(de_lines))]
            for idx, line in enumerate(de_lines):
                if get_sentence_len(line) <= TRIM_THRESH:
                    idx_mask[idx] = 1
            de_lines_new = list(compress(de_lines, idx_mask))
            en_lines_new = list(compress(en_lines, idx_mask))


    de_file_path.unlink()
    en_file_path.unlink()


    with open(de_file_path, mode='x') as df:
        with open(en_file_path, mode='x') as ef:
            df.writelines(de_lines_new)
            ef.writelines(en_lines_new)


if __name__ == '__main__':
    de_path_train = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/train.de-en.de'))
    en_path_train = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/train.de-en.en'))
    de_path_val = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/IWSLT16.TED.tst2013.de-en.de'))
    en_path_val = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/IWSLT16.TED.tst2013.de-en.en'))
    de_path_test = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/IWSLT16.TED.tst2014.de-en.de'))
    en_path_test = os.path.expanduser(Path('~/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/IWSLT16.TED.tst2014.de-en.en'))
    trim_lang_pair(Path(de_path_train), Path(en_path_train))
    trim_lang_pair(Path(de_path_val), Path(en_path_val))
    trim_lang_pair(Path(de_path_test), Path(en_path_test))
