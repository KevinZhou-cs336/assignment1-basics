import collections
import regex as re
import multiprocessing
from functools import partial

from typing import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries


class BPETokenizer(object):

    def train_bpe(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        num_cores = multiprocessing.cpu_count() - 1

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_cores, b"<|endoftext|>")
            segments = []

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                segments.append(f.read(end - start).decode("utf-8", errors="ignore"))

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            partial_func = partial(
                self.process_segment_pre_tokenization, special_tokens=special_tokens
            )
            # counts represent result from pre-tokenization for all chunks
            counts = {}
            with multiprocessing.Pool(processes=num_cores) as pool:
                counts = pool.map(partial_func, segments)

            # flatten counts as it contains multiple counter for different chunks
            tokenized_results = Counter()
            for count in counts:
                tokenized_results += count

            return self.process_bpe_merge(vocab_size, special_tokens, tokenized_results)

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
                    if pair_counts[old_pair] == 0:
                        pair_counts.pop(old_pair)
                    pair_to_token_keys[old_pair].discard(old_token)

                for new_pair in zip(new_token, new_token[1:]):
                    pair_counts[new_pair] += old_token_freq
                    pair_to_token_keys[new_pair].add(tuple(new_token))
            # clean up old pair to pre-token mapping
            pair_to_token_keys.pop(merge_pair)

        return vocab, merges

    def process_segment_pre_tokenization(
        self, segment: str, special_tokens: list[str]
    ) -> Counter:
        special_tokens_pattern = "|".join([re.escape(t) for t in special_tokens])

        counts = Counter()
        chunks = re.split(special_tokens_pattern, segment)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                token = match.group().encode("utf-8")
                res = []
                # convert b'how' to b'h, b'o', b'w'
                res = tuple([token[i : i + 1] for i in range(len(token))])
                counts[res] += 1

        return counts
