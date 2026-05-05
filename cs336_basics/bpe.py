import collections
import pickle
import regex as re
import multiprocessing
from functools import partial

from typing import Counter, Iterable
from cs336_basics.pretokenization_example import find_chunk_boundaries


class BPETokenizer(object):
    # pre-tokenize token
    PRE_TOKEN_PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.num_cores = multiprocessing.cpu_count() - 1
        self.vocab = vocab
        # map from vocab element to id
        self.inverted_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merges_rank = {m: idx for idx, m in enumerate(merges)}
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = (
                # sort special token to make sure longer speical token is matched first
                sorted(special_tokens, key=len, reverse=True)
            )
        self.special_tokens_set = set(self.special_tokens)

    @classmethod
    def from_file(cls, vocab_filepath, merges_filepath, special_tokens=[]):
        vocab, merges = None, None
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # 1. split by special tokens
        # 2. pre-tokenize
        # 3. process each token using merge rules
        """
        Optimized BPE encode using a linked list + priority queue:

        1. Build a merge rank dict: {(A, B): rank} for O(1) priority lookup.
        2. For each pre-token:
           a. Initialize a doubly linked list of individual bytes.
           b. For each adjacent pair in the token, if it exists in the merge rank dict,
              add (rank, pair_node) to a min-heap (priority queue).
           c. Repeatedly pop the lowest-rank (highest-priority) pair from the heap:
              - If the pair node is already marked as merged (lazy deletion), skip it.
              - Otherwise, merge the two nodes into one in the linked list.
              - Mark the two old neighbor pairs as merged (invalidate them).
              - Generate the new adjacent pairs with the merged node's neighbors,
                and push them onto the heap if they exist in the merge rank dict.
           d. Continue until the heap is empty.
           e. Traverse the linked list to collect the final token IDs.
        """
        if text is None:
            # deal with empty input
            return []
        if len(self.special_tokens) != 0:
            special_tokens_pattern = (
                "(" + "|".join([re.escape(t) for t in self.special_tokens]) + ")"
            )
            sentences = re.split(special_tokens_pattern, text)
        else:
            sentences = [text]

        encoded_text = []
        for sentence in sentences:
            # current sentence is a special token, encode it directly
            if sentence in self.special_tokens_set:
                encoded_text.append(self.inverted_vocab[sentence.encode("utf-8")])
                continue
            for match in BPETokenizer.PRE_TOKEN_PAT.finditer(sentence):
                token = match.group().encode("utf-8")
                token_elements = [token[i : i + 1] for i in range(len(token))]
                for merge in self.merges:
                    idx = 0
                    while idx < len(token_elements) - 1:
                        if (
                            token_elements[idx] != merge[0]
                            or token_elements[idx + 1] != merge[1]
                        ):
                            idx += 1
                            continue
                        token_elements = (
                            token_elements[:idx]
                            + [b"".join(merge)]
                            + token_elements[idx + 2 :]
                        )
                        idx += 1
                for element in token_elements:
                    encoded_text.append(self.inverted_vocab[element])

        return encoded_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for element in iterable:
            # yield will return entire list
            # yield from will return
            yield from self.encode(element)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")

    def train_bpe(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        chunks = self._split_file_to_chunks(input_path)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        partial_func = partial(self._pre_tokenize_chunk, special_tokens=special_tokens)
        # counts represent result from pre-tokenization for all chunks
        counts = {}
        with multiprocessing.Pool(processes=self.num_cores) as pool:
            counts = pool.map(partial_func, chunks)

        # flatten counts as it contains multiple counter for different chunks
        tokenized_results = Counter()
        for count in counts:
            tokenized_results += count

        return self.process_bpe_merge(vocab_size, special_tokens, tokenized_results)

    def _split_file_to_chunks(self, input_path: str) -> list[str]:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_cores, b"<|endoftext|>")
            chunks = []

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            return chunks

    def process_bpe_merge(
        self, vocab_size: int, special_tokens: list[str], tokenized_results: Counter
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab = {
            i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))
        }
        special_vocab_size = len(vocab)
        for i in range(256):
            vocab[i + special_vocab_size] = i.to_bytes(1)

        pair_counts = Counter()
        pair_to_token_keys = collections.defaultdict(set)
        # count pair freqs
        # (q, u) -> [question, quit]
        for pre_token, freq in tokenized_results.items():
            for pair in zip(pre_token, pre_token[1:]):
                pair_counts[pair] += freq
                pair_to_token_keys[pair].add(pre_token)

        merges = []
        while len(vocab) < vocab_size:
            # add new merge sequence
            merge_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges.append(merge_pair)

            # add new vocab
            vocab[len(vocab)] = b"".join(merge_pair)

            old_tokens = pair_to_token_keys[merge_pair]
            # process existing pairs, note use list to copy existing tokens to avoid
            # reading a collection(pair_to_token_keys) that is being updated
            for old_token in list(old_tokens):
                new_token = []
                old_token_freq = tokenized_results.pop(tuple(old_token))
                i = 0
                while i <= len(old_token) - 1:
                    if (
                        i < len(old_token) - 1
                        and old_token[i] == merge_pair[0]
                        and old_token[i + 1] == merge_pair[1]
                    ):
                        new_token.append(b"".join(merge_pair))
                        i += 2
                    else:
                        new_token.append(old_token[i])
                        i += 1
                tokenized_results[tuple(new_token)] += old_token_freq

                for old_pair in zip(old_token, old_token[1:]):
                    pair_counts[old_pair] -= old_token_freq
                    pair_to_token_keys[old_pair].discard(old_token)
                    # old pair doesn't exist anymore, remove all the records for the pair
                    if pair_counts[old_pair] == 0:
                        pair_counts.pop(old_pair)
                        pair_to_token_keys.pop(old_pair)

                for new_pair in zip(new_token, new_token[1:]):
                    pair_counts[new_pair] += old_token_freq
                    pair_to_token_keys[new_pair].add(tuple(new_token))
            # clean up old pair to pre-token mapping
            pair_to_token_keys.pop(merge_pair, None)

        return vocab, merges

    def _pre_tokenize_chunk(self, token: str, special_tokens: list[str]) -> Counter:
        special_tokens_pattern = "|".join([re.escape(t) for t in special_tokens])

        counts = Counter()
        chunks = re.split(special_tokens_pattern, token)
        for chunk in chunks:
            for match in BPETokenizer.PRE_TOKEN_PAT.finditer(chunk):
                token = match.group().encode("utf-8")
                res = []
                # convert b'how' to b'h, b'o', b'w'
                res = tuple([token[i : i + 1] for i in range(len(token))])
                counts[res] += 1

        return counts
