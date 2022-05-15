import torch
from .config import BATCH_SIZE, DEVICE


def train(model, iterator, iter_len, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for _, batch in enumerate(iterator):
        
        src = batch[0]
        trg = batch[1]

        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / (iter_len / BATCH_SIZE)


def evaluate(model, iterator, iter_len, criterion):    
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / (iter_len / BATCH_SIZE)
