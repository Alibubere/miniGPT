from collections import Counter
import json
import logging


class SimpleBPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self, text, vocab_size):

        processed_text = text.replace(" ", "Ġ")

        unique_chars = [chr(i) for i in range(256)]

        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: ch for i, ch in enumerate(unique_chars)}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}

        token_ids = [self.inverse_vocab[ch] for ch in text]

        logging.info(f"Starting training... Initial vocab size: {len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            pairs = Counter(zip(token_ids, token_ids[1:]))

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id

            i = 0
            new_tokens = []
            while i < len(token_ids):
                if (
                    i < len(token_ids) - 1
                    and (token_ids[i], token_ids[i + 1]) == best_pair
                ):
                    new_tokens.append(new_id)
                    i += 2

                else:
                    new_tokens.append(token_ids[i])
                    i += 1

            token_ids = new_tokens

            p0, p1 = best_pair
            merged = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged

            self.inverse_vocab[merged] = new_id

    def encode(self, text):

        text = text.replace(" ", "Ġ")
        token_ids = [self.inverse_vocab[ch] for ch in text]

        while True:
            pairs = [
                (token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)
            ]

            mergeable = [p for p in pairs if p in self.merges]

            if not mergeable:
                break

            best_pair = mergeable[0]
            new_id = self.merges[best_pair]

            i = 0

            new_tokens = []

            while i < len(token_ids):

                if (
                    i < len(token_ids) - 1
                    and (token_ids[i], token_ids[i + 1]) == best_pair
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            token_ids = new_tokens

        return new_tokens

    def decode(self, token_ids):
        text = "".join([self.vocab[i] for i in token_ids])

        return text.replace("Ġ", " ")

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):

        serializable_merges = {
            k: list(v) if isinstance(v, bytes) else v for k, v in self.merges.items()
        }

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(serializable_merges, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges.items()
            ]

            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):

        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)

            self.vocab = {int(k): v for k, v in loaded_vocab.items()}

            self.inverse_vocab = {v: int(k) for k, v in self.vocab.items()}

        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            self.merges = {}

            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.merges[pair] = new_id
