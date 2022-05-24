import torch
import spacy

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<bos>'] + tokens + ['eos']
    #print(tokens)
        
    src_indexes = [src_vocab.get_stoi().get(token, 0) for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_vocab.get_stoi()['<bos>']]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.get_stoi()['<eos>']:
            break
    
    trg_tokens = [trg_vocab.get_itos()[i] for i in trg_indexes]
    trg_tokens = ' '.join(trg_tokens[1:-1])
    
    return trg_tokens, attentions[:len(trg_tokens)-1]