from src.train import train_epoch, evaluate
import torch
from src.train.config import *
from src.preprocessing.config import *
from src.preprocessing.tokens_preprocessing import get_vocab_transforms
from src.models.transformer import Seq2SeqTransformer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse
import time
from src.train.utils import epoch_time
import math
from torchtext.datasets import Multi30k, IWSLT2016
from torch.utils.data import DataLoader
from src.preprocessing import collate_fn
from functools import partial
import torchdata.datapipes as dp

MODELS_PATH = Path('../../data/interim/transformer_runs')

parser = argparse.ArgumentParser(description='Transformer model training process wrapper')
parser.add_argument('epochs_num', type=int)
parser.add_argument('--ds_type', default='30k')
parser.add_argument('--resume', nargs='?', const=1, default=0)
parser.add_argument('--w_path')


def train_wrapper(last_epoch, epochs_num, model, train_dl, val_dl, optimizer, loss_fn):
    if train_len > 30000:
        ds_postfix = 'large'
    else:
        ds_postfix = 'default'
    run_name = f'{last_epoch}-{last_epoch + epochs_num}_{ds_postfix}'
    global_val_loss = float('inf')
    writer = SummaryWriter(MODELS_PATH / run_name)
    print(f'Starting from epoch: {last_epoch}')
    for epoch in range(last_epoch + 1, last_epoch + epochs_num + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_dl, train_len, optimizer, loss_fn, writer, epoch)
        print('Finished train epoch')
        val_loss = evaluate(model, val_dl, val_len, loss_fn)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        writer.add_scalars('Training vs validation loss', {'Training': train_loss, 'Validation': val_loss}, epoch)
        if val_loss < global_val_loss:
            global_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()},
                MODELS_PATH / run_name / 'transf_cp.tar')
            print('## Vall loss decreased, model succefully saved ##')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
    writer.flush()


if __name__ == '__main__':

    args = parser.parse_args()
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    if vars(args).get('ds_type') != '30k':
        train_iter += IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        val_iter += IWSLT2016(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))


    def get_sentence_len(line):
        return len(line.split())


    def filter_long_lines(line: str):
        de_line, en_line = line
        return get_sentence_len(de_line) < TRIM_THRESH and len(de_line) > 0


    train_iter = dp.iter.Filter(train_iter, filter_long_lines)
    val_iter = dp.iter.Filter(val_iter, filter_long_lines)

    # train_iter = dp.iter.Enumerator(dp.iter.Header(train_iter)) # TODO: bug in torchdata IterToMapDataPipeConverter
    # train_iter = dp.map.IterToMapConverter(train_iter)
    # val_iter = dp.iter.Enumerator(val_iter)
    # val_iter = dp.map.IterToMapConverter(val_iter)
    # train_len = len(train_iter)
    # val_len = len(val_iter)

    temp = [el for el in train_iter]  # shame, but dunno how to extract len from default ZipperIterDataPipe in another way
    train_len = len(temp) - 1

    temp = [el for el in val_iter]
    val_len = len(temp) - 1

    vocab_transform = get_vocab_transforms(train_iter)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, vocab_transform=vocab_transform), num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, vocab_transform=vocab_transform), num_workers=NUM_WORKERS)

    print(train_len, val_len)
    print(vars(args))

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    epochs_num = vars(args)['epochs_num']

    if vars(args).get('resume'):
        checkpoint = torch.load(vars(args).get('w_path'))
        epoch = checkpoint['epoch']
        transformer.load_state_dict(checkpoint['model_state_dict'])
        print('Resuming training proces...')
        train_wrapper(epoch, epochs_num, transformer, train_dataloader, val_dataloader, optimizer, loss_fn)
    else:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('Starting new training process...')
        train_wrapper(0, epochs_num, transformer, train_dataloader, val_dataloader, optimizer, loss_fn)
