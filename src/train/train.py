import torch
from tqdm import tqdm

from .config import DEVICE, BATCH_SIZE
from .utils import create_mask

def train_epoch(model, iterator, iter_len, optimizer, loss_fn, writer, epoch_num):
    model.train()
    losses = 0

    batches = list(iterator)
    index = epoch_num * len(batches)
    for src, tgt in tqdm(batches):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        tgt_out = tgt_out.to(torch.int64)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        writer.add_scalars('Batch loss', {'Training': loss.item()}, index)
        index += 1

    return losses / (iter_len / BATCH_SIZE)


def evaluate(model, iterator, iter_len, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in iterator:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / (iter_len / BATCH_SIZE)
