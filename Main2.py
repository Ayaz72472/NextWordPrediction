import pickle
import torch.nn as nn
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained LSTM model (weights from .h5 file, tokenizer from .pickle)
with open('tokenizer.pickle', 'rb') as file:
    lstm_tokenizer = pickle.load(file)

lstm_model = load_model('model_150.h5')  # Load the LSTM model

# Load the RNN model
rnn_model = tf.keras.models.load_model('final_model.keras')

# Define RNN-specific encoding function if needed
def rnn_encode(tokenizer, text_sentence, max_len=10):
    tokens = tokenizer.texts_to_sequences([text_sentence])
    padded_tokens = np.zeros((1, max_len))  # Adjust max_len as needed
    padded_tokens[:, :len(tokens[0])] = tokens[0][:max_len]
    mask_idx = text_sentence.split().index('[MASK]')
    return padded_tokens, mask_idx

# RNN decoding function
def rnn_decode(tokenizer, predictions, top_clean):
    top_predictions = np.argsort(predictions)[::-1][:top_clean]
    tokens = [tokenizer.index_word.get(idx, '') for idx in top_predictions]
    return '\n'.join(tokens)

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

from transformers import BartTokenizer, BartForConditionalGeneration
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

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
    
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    lstm_input, mask_idx = lstm_encode(lstm_tokenizer, text_sentence)
    predictions = lstm_model.predict(lstm_input)[0]
    lstm = lstm_decode(lstm_tokenizer, predictions[mask_idx], top_clean)
    print(f"LSTM Predictions: {lstm}")

    rnn_input, mask_idx = rnn_encode(lstm_tokenizer, text_sentence) 
    predictions = rnn_model.predict(rnn_input)[0]
    rnn = rnn_decode(lstm_tokenizer, predictions[mask_idx], top_clean)



    return {'bert': bert,
            'rnn': rnn,
            'bart': bart,
            'lstm': lstm}
