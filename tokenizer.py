import re
import copy

#Naive tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab_to_id):
        self.vocab_to_id = copy.deepcopy(vocab_to_id)
        self.id_to_vocab = {token: word for word, token in self.vocab_to_id.items()}
    
    def encode(self, text):
        """
        Tokenize text into list of tokens
        
        Args:
            text (str): input text
        
        Returns:
            list: list of token ids
        """
        #First we need to split the text according to whitespace and puncutation
        tokens = [word for word in re.split(r'([,.:;?_!"()\']|--|\s)', text) if word.split()]
        #Then we convert the tokens into token ids
        tokens = [self.vocab_to_id[token] for token in tokens]
        return tokens
    
    def decode(self, ids):
        #Decode the words and the punctuations
        text = " ".join([self.id_to_vocab[id] for id in ids])

        #Remove white spaces before the punctuations
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text)
        #Need to also remove the white spaces before and after --
        text = re.sub(r'\s+--\s+',r'--',text)
        return text
    

#Tokenizer with <|unk|> and <|endoftext|> tokens
class SimpleTokenizerV2:
    def __init__(self, vocab_to_id):
        #Test to check if the <|unk|> and <|endoftext|> tokens in the vocab
        self.vocab_to_id = copy.deepcopy(vocab_to_id)
        if not '<|unk|>' in self.vocab_to_id:
            self.vocab_to_id['<|unk|>'] = max(list(self.vocab_to_id.values())) + 1
        if not '<|endoftext|>' in self.vocab_to_id:
            #The reason why we can't just do vocab_to_id['<|unk|>'] is because in the case
            #that <|unk|> was present and somewhere in the middle of the text, we could potentially
            #overwrite a vocabulary value.
            self.vocab_to_id['<|endoftext|>'] = max(list(self.vocab_to_id.values())) + 1
        
        self.id_to_vocab = {token: word for word, token in self.vocab_to_id.items()}
    
    def encode(self, text):
        """
        Tokenize text into list of tokens
        
        Args:
            text (str): input text
        
        Returns:
            list: list of token ids
        """
        #First we need to split the text according to whitespace and puncutation
        tokens = [word for word in re.split(r'([,.:;?_!"()\']|--|\s)', text) if word.split()]
        #Then we convert the tokens into token ids
        tokens = [self.vocab_to_id[token] if token in self.vocab_to_id
                  else self.vocab_to_id['<|unk|>'] for token in tokens]
        return tokens
    
    def decode(self, ids):
        #Decode the words and the punctuations
        text = " ".join([self.id_to_vocab[id] for id in ids])

        #Remove white spaces before the punctuations
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text)
        #Need to also remove the white spaces before and after --
        text = re.sub(r'\s+--\s+',r'--',text)
        return text