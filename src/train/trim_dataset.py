from pathlib import Path

from tqdm import tqdm

de_path = Path(r'/home/user/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/train.de-en.de')
en_path = Path(r'/home/user/.torchtext/cache/IWSLT2016/2016-01/texts/de/en/de-en/train.de-en.en')


def get_sentence_len(line):
    return len(line.split())


def filter_idx(lines, idx_to_remove):
    for idx, line in tqdm(enumerate(lines)):
        if idx not in idx_to_remove:
            yield line


with open(de_path, mode='r') as de_file:
    with open(en_path, mode='r') as en_file:
        de_lines = de_file.readlines()
        en_lines = en_file.readlines()
        idx_to_remove = []

        for idx, line in enumerate(de_lines):
            if get_sentence_len(line) > 20:
                idx_to_remove.append(idx)

        # print(idx_to_remove)
        print(len(idx_to_remove))

        en_lines_new = list(filter_idx(en_lines, idx_to_remove))
        de_lines_new = list(filter_idx(en_lines, idx_to_remove))

        print(len(en_lines_new))
        print(len(de_lines_new))

de_path.unlink()
en_path.unlink()

with open(de_path, mode='x') as de_file:
    with open(en_path, mode='x') as en_file:
        de_file.writelines(de_lines_new)
        en_file.writelines(en_lines_new)

#     lines = file.readlines()
#     print(max([get_sentence_len(line) for line in lines]))
#     print(sum([get_sentence_len(line) for line in lines]) / len(lines))
