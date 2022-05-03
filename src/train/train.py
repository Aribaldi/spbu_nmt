import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k, IWSLT2016
from .config import DEVICE, BATCH_SIZE, NUM_WORKERS, TRAIN_LEN, VAL_LEN
from src.preprocessing import SRC_LANGUAGE, TGT_LANGUAGE, collate_fn
from .utils import create_mask


def train_epoch(model, optimizer, loss_fn, dataset_type='30k'):
    model.train()
    losses = 0
    if dataset_type == 'both':
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) + \
                    IWSLT2016(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    elif dataset_type == '30k':
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        tgt_out = tgt_out.to(torch.int64)
        #logits = logits.to(torch.int64)
        try:
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        except:
            print('error')
            print(logits.dtype,tgt_out.dtype)
            #print(logits, tgt_out) 
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / ( TRAIN_LEN / BATCH_SIZE)


def evaluate(model, loss_fn, dataset_type='30k'):
    model.eval()
    losses = 0
    if dataset_type == 'both':
        val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) + \
                IWSLT2016(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    elif dataset_type == '30k':
        val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / ( VAL_LEN / BATCH_SIZE)