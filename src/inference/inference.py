from torchtext.datasets import Multi30k, IWSLT2016

from src.models.transformer import Seq2SeqTransformer
from src.preprocessing.config import *
from src.preprocessing.tokens_preprocessing import get_vocab_transforms
from src.train.config import *
from src.train.utils import generate_square_subsequent_mask
from src.preprocessing.collation import text_transform


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str, vocab_transform, text_transform):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#train_iter += IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_iter = list(iter(train_iter))

test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#test_iter += IWSLT2016(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test_iter = list(iter(test_iter))

vocab_transform = get_vocab_transforms(train_iter)
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
checkpoint = torch.load('../../data/interim/transformer_runs/0-20_default/transf_cp.tar')
epoch = checkpoint['epoch']
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer = transformer.to(DEVICE)


for i in range(0, 50):
    sentence = test_iter[i]
    print(f'Sentence: \"{sentence}\"')
    print(translate(transformer, sentence[0], vocab_transform, text_transform(vocab_transform)))
