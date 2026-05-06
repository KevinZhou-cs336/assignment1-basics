from __future__ import annotations
from typing import Counter, Iterable

import collections
import heapq
import pickle
import regex as re
import multiprocessing
from functools import partial

from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries


class IndividualBytes(object):
    def __init__(
        self,
        idx: int,
        val: bytes,
        prev: IndividualBytes = None,
        next: IndividualBytes = None,
        merged: bool = False,
    ):
        self.idx = idx
        self.val = val
        self.prev = prev
        self.next = next
        self._merged = merged

    def merge(self):
        self._merged = True

    def is_merged(self):
        return self._merged


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
    def from_file(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
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
                encoded_token = self._encode_single_token(match.group())
                encoded_text.extend(encoded_token)

        return encoded_text

    def _encode_single_token(self, token: str) -> list[int]:
        token_bytes = token.encode("utf-8")
        head, _ = self._build_linked_individual_bytes_from_token(token_bytes)
        # merge queue doesn't contain the head, tail sentinel node
        merge_queue = self._initialize_token_merge_queue(head)

        while merge_queue:
            _, _, existing_pair, cur_node = heapq.heappop(merge_queue)
            left_node = cur_node.prev
            right_node = cur_node.next
            # current node or right node is already merged, this is an obsolete merge pair
            if cur_node.is_merged() or right_node.is_merged():
                continue

            cur_pair = (cur_node.val, right_node.val)
            if cur_pair != existing_pair:
                continue

            # merge cur node and the right node since we are merged these two individual bytes
            right_node.merge()
            merged_elem = b"".join(cur_pair)
            cur_node.val = merged_elem
            right_node = right_node.next
            cur_node.next = right_node
            right_node.prev = cur_node
            right_pair = (merged_elem, right_node.val)
            if not right_node.is_merged() and right_pair in self.merges_rank:
                heapq.heappush(
                    merge_queue,
                    (self.merges_rank[right_pair], cur_node.idx, right_pair, cur_node),
                )
            left_pair = (left_node.val, merged_elem)
            if not left_node.is_merged() and left_pair in self.merges_rank:
                heapq.heappush(
                    merge_queue,
                    (self.merges_rank[left_pair], left_node.idx, left_pair, left_node),
                )

        encoded_token = []
        # start from non-sentinel(head) pair
        cur = head
        while cur:
            if cur.is_merged():
                cur = cur.next
                continue
            encoded_token.append(self.inverted_vocab[cur.val])
            cur = cur.next

        return encoded_token

    def _build_linked_individual_bytes_from_token(
        self, token: bytes
    ) -> tuple[IndividualBytes, IndividualBytes]:
        head, tail = IndividualBytes(-1, b"", merged=True), IndividualBytes(
            -1, b"", merged=True
        )
        token_elements = [token[i : i + 1] for i in range(len(token))]
        cur_node = head
        for idx, element in enumerate(token_elements):
            new_node = IndividualBytes(idx, element)
            cur_node.next = new_node
            new_node.prev = cur_node
            cur_node = new_node
        cur_node.next = tail
        tail.prev = cur_node

        return head, tail

    def _initialize_token_merge_queue(self, cur: IndividualBytes) -> IndividualBytes:
        heap = []
        # skip the head sentinel pair
        cur = cur.next
        while not cur.next.is_merged():
            cur_pair = (cur.val, cur.next.val)
            if cur_pair not in self.merges_rank:
                cur = cur.next
                continue
            heapq.heappush(heap, (self.merges_rank[cur_pair], cur.idx, cur_pair, cur))
            cur = cur.next

        return heap

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
        with multiprocessing.Pool(processes=self.num_cores) as pool:
            counts = pool.map(partial_func, chunks)

        # flatten counts as it contains multiple counter for different chunks
        tokenized_results = Counter()
        for count in counts:
            tokenized_results += count

        return self._process_bpe_merge(vocab_size, special_tokens, tokenized_results)

    def _split_file_to_chunks(self, input_path: str) -> list[str]:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_cores, b"<|endoftext|>")
            chunks = []

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            return chunks

    def _process_bpe_merge(
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
                while i < len(old_token):
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

    def _pre_tokenize_chunk(self, text_chunk: str, special_tokens: list[str]) -> Counter:
        special_tokens_pattern = "|".join([re.escape(t) for t in special_tokens])

        counts = Counter()
        chunks = re.split(special_tokens_pattern, text_chunk)
        for chunk in chunks:
            for match in BPETokenizer.PRE_TOKEN_PAT.finditer(chunk):
                token_bytes = match.group().encode("utf-8")
                # convert b'how' to (b'h', b'o', b'w')
                res = tuple([token_bytes[i : i + 1] for i in range(len(token_bytes))])
                counts[res] += 1

        return counts
