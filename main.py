# %%
import torch
import string
import pickle
import torch.nn as nn
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained LSTM model (weights from .h5 file, tokenizer from .pickle)
with open('tokenizer.pickle', 'rb') as file:
    lstm_tokenizer = pickle.load(file)

lstm_model = load_model('model_150.h5')  # Load the LSTM model

# Ensure the LSTM model is ready for inference
lstm_model.summary()
def lstm_encode(tokenizer, text_sentence, max_len=10):
    """
    Encode the input sentence for the LSTM model.
    This will tokenize and pad sequences to the expected input shape of the LSTM model.
    """
    text_sentence = text_sentence.replace('<mask>', '[MASK]')
    tokens = tokenizer.texts_to_sequences([text_sentence])
    padded_tokens = np.zeros((1, max_len))  # Adjust max_len to match your LSTM input size
    padded_tokens[:, :len(tokens[0])] = tokens[0][:max_len]  # Pad or truncate
    mask_idx = text_sentence.split().index('[MASK]')
    return padded_tokens, mask_idx


def lstm_decode(tokenizer, predictions, top_clean):
    """
    Decode the top predictions into readable tokens.
    """
    top_predictions = np.argsort(predictions)[::-1][:top_clean]  # Get top predictions
    tokens = [tokenizer.index_word.get(idx, '') for idx in top_predictions]
    print("Decoded LSTM Tokens: ", tokens)  # Add a log to check the output
    return '\n'.join(tokens)



from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

from transformers import XLNetTokenizer, XLNetLMHeadModel
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

from transformers import BartTokenizer, BartForConditionalGeneration
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

from transformers import ElectraTokenizer, ElectraForMaskedLM
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLNET LARGE =================================
    input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    with torch.no_grad():
        predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLM ROBERTA BASE =================================
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= BART =================================
    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ELECTRA =================================
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ROBERTA =================================
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    # ========================= ROBERTA =================================
    lstm_input, mask_idx = lstm_encode(lstm_tokenizer, text_sentence)
    predictions = lstm_model.predict(lstm_input)[0]
    lstm = lstm_decode(lstm_tokenizer, predictions[mask_idx], top_clean)
    print(f"LSTM Predictions: {lstm}")


    return {'bert': bert,
            'xlnet': xlnet,
            'xlm': xlm,
            'bart': bart,
            'electra': electra,
            'roberta': roberta,
            'lstm': lstm}
