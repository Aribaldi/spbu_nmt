import torch
from src.train.config import *
from src.preprocessing.config import *
from src.preprocessing.tokens_preprocessing import get_vocab_transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim
import time
from torch import optim
import warnings
warnings.filterwarnings('ignore')

from torchtext.datasets import Multi30k, IWSLT2016
from torch.utils.data import DataLoader
from src.preprocessing import collate_fn

from src.models.seq2seq import Attention, Encoder, Decoder, Seq2Seq
from src.train.seq2seq_train import train, evaluate
from src.train.utils import epoch_time
import math
import argparse
from pathlib import Path
from functools import partial


MODELS_PATH = Path('../../data/interim/s2s_runs')

parser = argparse.ArgumentParser(description='Seq2Seq model training process wrapper')
parser.add_argument('epochs_num', type=int)
parser.add_argument('--ds_type', default='30k')
parser.add_argument('--resume', nargs='?', const=1, default=0)
parser.add_argument('--w_path')


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train_wrapper(last_epoch, epochs_num, model, train_dl, val_dl, optimizer, criterion, clip):
    run_name = f'{last_epoch}-{last_epoch + epochs_num}'
    best_valid_loss = float('inf')
    writer = SummaryWriter(MODELS_PATH / run_name)
    print(f'Starting from epoch: {last_epoch}')
    for epoch in range(last_epoch + 1, last_epoch + epochs_num + 1):
        start_time = time.time()
        train_loss = train(model, train_dl, train_len, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_dl, val_len, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        writer.add_scalars('Training vs validation loss', {'Training': train_loss, 'Validation': valid_loss}, epoch)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'model':model.state_dict(),
                        'epoch': epoch},
                        MODELS_PATH / run_name / 'seq2seq.tar')
            print('## Vall loss decreased, model successfully saved ##')
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    writer.flush()


if __name__ == '__main__':

    args = parser.parse_args()
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))


    if vars(args).get('ds_type'):
        train_iter += IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        val_iter += IWSLT2016(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    temp = [el for el in train_iter] #shame, but dunno how to extract len from default ZipperIterDataPipe in another way
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


    attn = Attention(S2S_ENC_HID_DIM, S2S_DEC_HID_DIM)
    enc = Encoder(SRC_VOCAB_SIZE, S2S_ENC_EMB_DIM, S2S_ENC_HID_DIM, S2S_DEC_HID_DIM, S2S_ENC_DROPOUT)
    dec = Decoder(TGT_VOCAB_SIZE, S2S_DEC_EMB_DIM, S2S_ENC_HID_DIM, S2S_DEC_HID_DIM, S2S_DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE, PAD_IDX).to(DEVICE)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    epochs_num = vars(args)['epochs_num']

    if vars(args).get('resume'):
        checkpoint = torch.load(vars(args).get('w_path'))
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print('Resuming training proces...')
        train_wrapper(epoch, epochs_num, model, train_dataloader, val_dataloader, optimizer, criterion, 1)
    else:
        model.apply(init_weights)
        print('Starting new training process...')
        train_wrapper(0, epochs_num, model, train_dataloader, val_dataloader, optimizer, criterion, 1)   