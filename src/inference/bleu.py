from torchtext.data.metrics import bleu_score
from .seq2seq_inference import translate_sentence
from src.inference.inference import translate
from tqdm import tqdm

def calculate_bleu(data, src_vocab, trg_vocab, model, device, iter_len, text_transform = None, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for sample in tqdm(data, total=iter_len):
        
        src = sample[0]
        trg = sample[1]
        
        if model.__class__.__name__ == 'Seq2Seq':
            pred_trg, _ = translate_sentence(src, src_vocab, trg_vocab, model, device, max_len)
        else:
            pred_trg = translate(model, src, trg_vocab, text_transform)
        pred_trg = pred_trg.split()
        trg = trg.split()

        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs, max_n=3, weights=[0.33, 0.33, 0.33])


