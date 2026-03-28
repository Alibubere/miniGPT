from collections import Counter

class SimpleBPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self,text,vocab_size):
        text = text.replace(" ","Ġ")

        char = sorted(list(set(text)))
        self.vocab = {i:ch for i , ch in enumerate(char)}
        self.inverse_vocab = {ch:i for i , ch in self.vocab.items()}

        token_ids = [self.inverse_vocab[ch] for ch in text]

        while len(self.vocab) < vocab_size:
            pairs = Counter(zip(token_ids,token_ids[1:]))

            if not pairs:
                break

            best_pair = max(pairs,key=pairs.get)
            new_id =  len(self.vocab)
            self.merges[best_pair] = new_id

            i = 0
            new_tokens = []
            while i < len(token_ids):
                if i <len(token_ids)-1 and (token_ids[i],token_ids[i+1]) == best_pair:
                    new_tokens.append(new_id)
                    i +=2

                else:
                    new_tokens.append(token_ids[i])
                    i +=1

            token_ids = new_tokens

            p0 , p1 = best_pair
            merged = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged

            self.inverse_vocab[merged] = new_id

    def encode(self,text):

        text = text.replace(" ","Ġ")
        token_ids = [self.inverse_vocab[ch] for ch in text] 

        while True:
            pairs = [(token_ids[i],token_ids[i+1]) for i in range(len(token_ids)-1)]

            mergeable = [p for p in pairs if p in self.merges]

            if not mergeable:
                break

            best_pair = mergeable[0]
            new_id = self.merges[best_pair]

            i= 0

            new_tokens = []

            while i <len(token_ids):

                if i < len(token_ids)-1 and (token_ids[i],token_ids[i+1]) == best_pair:
                    new_tokens.append(new_id)
                    i+= 2
                else:
                    new_tokens.append(token_ids[i])
                    i+=1
            token_ids = new_tokens

        return new_tokens
    
    def decode(self,token_ids):
        text = "".join([self.vocab[i] for i in token_ids])
        
        return text.replace("Ġ"," ")