import nltk

def normalize_sentence(sentence):
    return nltk.tokenize.word_tokenize(sentence.lower())


def translate_from_ids_to_text(ids, tokenizer):
    text = tokenizer.decode(ids)
    if '</s>' in text:
        text, pad = text.split('</s>', 1)
    if '<s>' in text:
        text = text[3:]
    
    #text_as_list = text.split(' ')
    return text