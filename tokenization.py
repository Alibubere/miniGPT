from collections import Counter
import json
import logging


class SimpleBPE:
    """
    A simple Byte Pair Encoding (BPE) tokenizer.

    Trains on raw text by iteratively merging the most frequent
    adjacent token pairs until the desired vocabulary size is reached.

    Attributes:
        vocab (dict[int, str]): Maps token ID to string representation.
        inverse_vocab (dict[str, int]): Maps string representation to token ID.
        merges (dict[tuple[int, int], int]): Maps merged pair to new token ID.
    """

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train the BPE tokenizer on the given text.

        Initializes vocabulary with all 256 ASCII characters plus any
        extra characters found in the text, then iteratively merges
        the most frequent adjacent token pairs until vocab_size is reached.

        Args:
            text (str): Raw training text.
            vocab_size (int): Target vocabulary size. Must be >= 256.
        """
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

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.

        Applies learned BPE merges in order to the input text.
        Spaces are replaced with 'Ġ' before encoding.

        Args:
            text (str): Raw input text to encode.

        Returns:
            list[int]: Sequence of token IDs.
        """

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

    def decode(self, token_ids: list[int]):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (list[int]): Sequence of token IDs to decode.

        Returns:
            str: Reconstructed text with 'Ġ' replaced by spaces.
        """

        text = "".join([self.vocab[i] for i in token_ids])

        return text.replace("Ġ", " ")

    def save_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Persist the vocabulary and merge rules to disk as JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary JSON file.
            bpe_merges_path (str): Path to save the BPE merges JSON file.
        """

        serializable_vocab = {
            k: list(v) if isinstance(v, bytes) else v for k, v in self.vocab.items()
        }

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(serializable_vocab, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges.items()
            ]

            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Load vocabulary and merge rules from saved JSON files.

        Populates self.vocab, self.inverse_vocab, and self.merges.

        Args:
            vocab_path (str): Path to the saved vocabulary JSON file.
            bpe_merges_path (str): Path to the saved BPE merges JSON file.
        """

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
