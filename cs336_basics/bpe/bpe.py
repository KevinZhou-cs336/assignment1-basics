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
    """A node in a doubly linked list representing one byte-chunk during BPE encoding.

    Why a doubly linked list?
        BPE encoding repeatedly merges adjacent pairs of byte-chunks. After a merge,
        the merged node's NEW neighbours must be checked for further merges. With a
        plain list this would require O(n) scanning; a doubly linked list lets us
        jump directly to the left and right neighbours in O(1), then update the
        pointers in O(1) — making the overall encode much faster.

    Lazy deletion — the `merged` flag:
        When two nodes are merged, the RIGHT node is marked merged=True and
        logically removed from the list (the left node absorbs its bytes).
        We don't physically delete nodes because the priority queue (heap) may
        still hold references to them. Instead, when a heap entry is popped we
        check is_merged() and discard stale entries immediately.

    Fields:
        idx:    Stable integer index (position in the original byte sequence).
                Used as a tie-breaker in the heap so merges at earlier positions
                win when two pairs have the same rank.
        val:    The bytes this node currently represents (grows as merges happen).
        prev:   Pointer to the left neighbour (or sentinel head if at start).
        next:   Pointer to the right neighbour (or sentinel tail if at end).
        merged: True once this node has been absorbed into its left neighbour.
    """

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
    """Byte-Pair Encoding (BPE) tokenizer (Sennrich et al., 2016).

    ── What is BPE? ────────────────────────────────────────────────────────────

    BPE is the tokenization algorithm used by GPT-2, GPT-3, LLaMA, and most
    modern language models. It converts raw text into a sequence of integer IDs
    that the model can process.

    The core idea: start with individual bytes (256 possible values), then
    repeatedly find the most frequent adjacent pair of symbols and merge them
    into a single new symbol. After many merges, common words and word-pieces
    become single tokens, while rare words are split into smaller pieces.

    Example (simplified):
        Corpus: "low low low lowest lowest newer newer newer"
        Initial tokens: [l,o,w], [l,o,w], ..., [n,e,w,e,r], ...
        Most frequent pair: (e, r) → merge into "er"
        Next most frequent pair: (l, o) → merge into "lo"
        ... continue until vocab_size is reached.

    ── Three-step pipeline ─────────────────────────────────────────────────────

    1. PRE-TOKENISATION: Split text into coarse "pre-tokens" using a regex.
       This prevents merges from crossing word boundaries (e.g., "dog." should
       not merge the period into the word token "dog."). Each pre-token is then
       treated independently.

    2. TRAINING (train_bpe): Count pair frequencies across all pre-tokens,
       greedily merge the most frequent pair, update counts, repeat until the
       vocabulary reaches vocab_size. Records the ordered merge list.

    3. ENCODING (encode): Given new text and the learned merge list, apply the
       same merges in the same priority order to tokenize the text into IDs.

    ── Key data structures ─────────────────────────────────────────────────────

    vocab:         dict[int → bytes]  maps token ID → byte string.
                   IDs 0..len(special_tokens)-1 are special tokens.
                   IDs len(special)..len(special)+255 are single raw bytes.
                   IDs above that are learned merge tokens.

    merges:        list[tuple[bytes, bytes]]  the ordered merge rules, e.g.
                   [(b'e', b'r'), (b'lo', b'w'), ...].  Order matters: earlier
                   merges have higher priority during encoding.

    merges_rank:   dict[(bytes, bytes) → int]  inverts the merges list for
                   O(1) priority lookup: rank 0 = highest priority merge.

    inverted_vocab: dict[bytes → int]  inverts vocab for O(1) ID lookup during encode.

    special_tokens: list[str]  tokens like "<|endoftext|>" that are matched
                   verbatim and never split by the regex or BPE algorithm.
                   Sorted longest-first so longer tokens match before shorter prefixes.
    """

    # Pre-tokenisation regex (GPT-2 style).
    # Splits text into coarse tokens BEFORE BPE merges are applied, ensuring
    # that merges never cross natural word/punctuation boundaries.
    #
    # Pattern breakdown (matched left-to-right, first match wins):
    #   '(?:[sdmt]|ll|ve|re)  — English contractions: 's, 'd, 'm, 't, 'll, 've, 're
    #   | ?\p{L}+             — optional space then one-or-more Unicode letters (a word)
    #   | ?\p{N}+             — optional space then one-or-more Unicode digits (a number)
    #   | ?[^\s\p{L}\p{N}]+  — optional space then one-or-more punctuation/symbols
    #   |\s+(?!\S)            — trailing whitespace (whitespace not followed by non-whitespace)
    #   |\s+                  — any remaining whitespace
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
        # Inverted vocab for O(1) bytes→ID lookup during encode.
        self.inverted_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        # Rank table: lower rank = higher priority = applied first during encode.
        self.merges_rank = {m: idx for idx, m in enumerate(merges)}
        if special_tokens is None:
            self.special_tokens = []
        else:
            # Sort longest-first: "endoftext" must match before "end" if both are special.
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_set = set(self.special_tokens)

    @classmethod
    def from_file(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        """Load a previously trained tokenizer from pickled vocab and merges files."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs.

        High-level algorithm:
          1. Split text on special tokens (e.g. "<|endoftext|>") so they are
             never broken up by the regex or BPE algorithm.
          2. For each non-special segment, run the pre-tokenisation regex to
             get coarse word/punctuation chunks.
          3. Apply BPE merges to each chunk independently via _encode_single_token.
          4. Collect all IDs in order.

        Optimised BPE encode using a linked list + priority queue:

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
            return []

        # Split on special tokens so they are encoded as a single ID each.
        # re.split with a capture group keeps the delimiters in the result list.
        if len(self.special_tokens) != 0:
            special_tokens_pattern = (
                "(" + "|".join([re.escape(t) for t in self.special_tokens]) + ")"
            )
            sentences = re.split(special_tokens_pattern, text)
        else:
            sentences = [text]

        encoded_text = []
        for sentence in sentences:
            # Special tokens map directly to a single ID — no BPE needed.
            if sentence in self.special_tokens_set:
                encoded_text.append(self.inverted_vocab[sentence.encode("utf-8")])
                continue
            # For normal text: split into pre-tokens and BPE-encode each one.
            for match in BPETokenizer.PRE_TOKEN_PAT.finditer(sentence):
                encoded_token = self._encode_single_token(match.group())
                encoded_text.extend(encoded_token)

        return encoded_text

    def _encode_single_token(self, token: str) -> list[int]:
        """Apply BPE merges to a single pre-token string and return token IDs.

        The token is first UTF-8 encoded into raw bytes, then represented as a
        doubly linked list of single-byte nodes. A min-heap drives the merge
        order: the pair with the lowest rank (= earliest in the merge list) is
        always processed next, matching the order used during training.

        Lazy deletion handles stale heap entries: when a node is merged away,
        its flag is set to True; any heap entry pointing to a merged node is
        simply skipped when popped, without needing to remove it from the heap.
        """
        token_bytes = token.encode("utf-8")
        head, _ = self._build_linked_individual_bytes_from_token(token_bytes)
        # Build the initial heap of all valid adjacent pairs.
        merge_queue = self._initialize_token_merge_queue(head)

        while merge_queue:
            # Pop the highest-priority (lowest rank) merge candidate.
            # Heap entry format: (rank, node_idx, pair_bytes, left_node)
            _, _, existing_pair, cur_node = heapq.heappop(merge_queue)
            right_node = cur_node.next
            left_node = cur_node.prev

            # Lazy deletion: skip if either node was already merged by a prior step.
            if cur_node.is_merged() or right_node.is_merged():
                continue

            # Also skip if the pair stored in the heap entry no longer matches
            # the current node values (node was merged and its val changed).
            cur_pair = (cur_node.val, right_node.val)
            if cur_pair != existing_pair:
                continue

            # ── Perform the merge ────────────────────────────────────────────
            # The right node is absorbed into the left (cur) node:
            #   - right_node.merge() marks it logically deleted
            #   - cur_node.val grows to include the right node's bytes
            #   - linked list pointers skip over right_node
            right_node.merge()
            merged_elem = b"".join(cur_pair)
            cur_node.val = merged_elem
            right_node = right_node.next  # advance past the now-deleted node
            cur_node.next = right_node
            right_node.prev = cur_node

            # Push the two NEW adjacent pairs created by the merge onto the heap.
            # (cur_node, right_node) — the merged node paired with its new right neighbour
            right_pair = (merged_elem, right_node.val)
            if not right_node.is_merged() and right_pair in self.merges_rank:
                heapq.heappush(
                    merge_queue,
                    (self.merges_rank[right_pair], cur_node.idx, right_pair, cur_node),
                )
            # (left_node, cur_node) — the merged node paired with its left neighbour
            left_pair = (left_node.val, merged_elem)
            if not left_node.is_merged() and left_pair in self.merges_rank:
                heapq.heappush(
                    merge_queue,
                    (self.merges_rank[left_pair], left_node.idx, left_pair, left_node),
                )

        # Traverse the list, skipping merged (deleted) nodes, to collect final IDs.
        encoded_token = []
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
        """Build a doubly linked list from a byte string.

        Structure: [sentinel_head] ↔ [byte_0] ↔ [byte_1] ↔ ... ↔ [byte_n] ↔ [sentinel_tail]

        The head and tail sentinels (both pre-marked as merged=True) simplify
        boundary checks: every real node always has a non-None prev and next,
        so we never need special-case logic for the first or last element.

        Returns: (head_sentinel, tail_sentinel)
        """
        head = IndividualBytes(-1, b"", merged=True)
        tail = IndividualBytes(-1, b"", merged=True)
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

    def _initialize_token_merge_queue(self, cur: IndividualBytes) -> list:
        """Build the initial min-heap of valid adjacent byte pairs.

        Only pairs that exist in merges_rank are added — pairs not in any merge
        rule will never be merged, so there is no reason to track them.

        Heap entry: (rank, node_idx, pair_bytes, left_node)
            rank:      merge priority (lower = higher priority)
            node_idx:  tie-breaker so that leftmost pairs win on equal rank
            pair_bytes: snapshot of (left.val, right.val) at insertion time;
                        compared at pop time to detect stale entries
            left_node: pointer to the left node of the pair
        """
        heap = []
        cur = cur.next  # skip the head sentinel
        while not cur.next.is_merged():  # stop before the tail sentinel
            cur_pair = (cur.val, cur.next.val)
            if cur_pair in self.merges_rank:
                heapq.heappush(heap, (self.merges_rank[cur_pair], cur.idx, cur_pair, cur))
            cur = cur.next

        return heap

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Lazily encode a stream of strings, yielding token IDs one by one.

        Uses `yield from` so each string is encoded and its IDs are yielded
        immediately, without accumulating the entire output in memory first.
        Useful for encoding large files line-by-line.
        """
        for element in iterable:
            yield from self.encode(element)

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string.

        Each ID is looked up in vocab to get its bytes, all byte strings are
        concatenated, then decoded as UTF-8. `errors='replace'` substitutes the
        Unicode replacement character (U+FFFD) for any byte sequences that are
        not valid UTF-8, which can happen when a multi-byte character is split
        across the boundary of two decoded chunks.
        """
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")

    def train_bpe(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE on a text file and return the learned vocab and merge list.

        Pipeline:
          1. Split file into chunks — one per CPU core — so pre-tokenisation
             can run in parallel (multiprocessing.Pool).
          2. Each worker pre-tokenises its chunk and returns a Counter mapping
             (byte_tuple) → frequency.
          3. Merge all per-chunk counters into one global frequency table.
          4. Run the BPE merge loop until vocab reaches vocab_size.

        Returns:
            vocab:  dict[int → bytes] mapping token IDs to their byte strings.
            merges: list of (bytes, bytes) merge rules in priority order.
        """
        chunks = self._split_file_to_chunks(input_path)

        # Pre-tokenise all chunks in parallel; partial() binds special_tokens into the worker fn.
        partial_func = partial(self._pre_tokenize_chunk, special_tokens=special_tokens)
        with multiprocessing.Pool(processes=self.num_cores) as pool:
            counts = pool.map(partial_func, chunks)

        # Combine per-chunk counters into one global pre-token frequency table.
        tokenized_results = Counter()
        for count in counts:
            tokenized_results += count

        return self._process_bpe_merge(vocab_size, special_tokens, tokenized_results)

    def _split_file_to_chunks(self, input_path: str) -> list[str]:
        """Split a text file into num_cores chunks, each ending at a document boundary.

        Why split at document boundaries?
            BPE should not merge bytes that span two separate documents.
            find_chunk_boundaries ensures every chunk ends at a "<|endoftext|>"
            token, so pre-tokens never straddle document boundaries.

        Returns a list of decoded text strings, one per chunk.
        """
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
        """Core BPE training loop: greedily merge the most frequent pair until vocab_size.

        Initial vocabulary:
            IDs 0 .. len(special_tokens)-1  → special tokens (e.g. "<|endoftext|>")
            IDs len(special) .. len(special)+255 → all 256 single raw bytes (0x00–0xFF)

        This guarantees the tokenizer can encode ANY byte sequence, even if a
        byte never appeared in training (it simply stays as a 1-byte token).

        Auxiliary data structures for O(1) updates:
            pair_counts:        Counter  {(A, B) → total occurrences across all pre-tokens}
            pair_to_token_keys: dict     {(A, B) → set of pre-tokens containing this pair}

        Why track pair_to_token_keys?
            After merging pair (A, B) into AB, we must:
              - Decrement counts for all pairs that contained A or B (now obsolete).
              - Increment counts for the new pairs (left_neighbour, AB) and (AB, right_neighbour).
            pair_to_token_keys tells us exactly which pre-tokens to update, so we
            avoid re-scanning the entire corpus on every merge step.

        Merge loop:
            Each iteration: find the most frequent pair (ties broken lexicographically),
            record it in `merges`, add the merged bytes to vocab, then update
            pair_counts and pair_to_token_keys to reflect the change.
        """
        # Build initial vocab: special tokens first, then all 256 single bytes.
        vocab = {
            i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))
        }
        special_vocab_size = len(vocab)
        for i in range(256):
            vocab[i + special_vocab_size] = i.to_bytes(1)

        # Build pair frequency table and reverse index from the pre-tokenised corpus.
        # tokenized_results: {(b'h', b'e', b'l', b'l', b'o') → 42, ...}
        pair_counts = Counter()
        pair_to_token_keys = collections.defaultdict(set)
        for pre_token, freq in tokenized_results.items():
            for pair in zip(pre_token, pre_token[1:]):
                pair_counts[pair] += freq
                pair_to_token_keys[pair].add(pre_token)

        merges = []
        while len(vocab) < vocab_size:
            # Select the most frequent pair; break ties lexicographically for determinism.
            merge_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges.append(merge_pair)

            # Add the merged symbol to the vocabulary.
            vocab[len(vocab)] = b"".join(merge_pair)

            # Update every pre-token that contained this pair.
            # We copy the set to a list because we modify pair_to_token_keys inside the loop.
            old_tokens = pair_to_token_keys[merge_pair]
            for old_token in list(old_tokens):
                # Apply the merge to this pre-token, producing a new (shorter) token.
                new_token = []
                old_token_freq = tokenized_results.pop(tuple(old_token))
                i = 0
                while i < len(old_token):
                    if (
                        i < len(old_token) - 1
                        and old_token[i] == merge_pair[0]
                        and old_token[i + 1] == merge_pair[1]
                    ):
                        # Merge: replace the two symbols with the new merged symbol.
                        new_token.append(b"".join(merge_pair))
                        i += 2
                    else:
                        new_token.append(old_token[i])
                        i += 1
                tokenized_results[tuple(new_token)] += old_token_freq

                # Decrement counts for all pairs in the OLD pre-token (now obsolete).
                for old_pair in zip(old_token, old_token[1:]):
                    pair_counts[old_pair] -= old_token_freq
                    pair_to_token_keys[old_pair].discard(old_token)
                    if pair_counts[old_pair] == 0:
                        pair_counts.pop(old_pair)
                        pair_to_token_keys.pop(old_pair)

                # Increment counts for all pairs in the NEW pre-token.
                for new_pair in zip(new_token, new_token[1:]):
                    pair_counts[new_pair] += old_token_freq
                    pair_to_token_keys[new_pair].add(tuple(new_token))

            # Remove the now-consumed merge pair from the reverse index.
            pair_to_token_keys.pop(merge_pair, None)

        return vocab, merges

    def _pre_tokenize_chunk(self, text_chunk: str, special_tokens: list[str]) -> Counter:
        """Pre-tokenise one text chunk and count byte-tuple frequencies.

        Why pre-tokenise?
            Raw text must be split into coarse "words" before BPE so that merges
            never cross word/punctuation boundaries. For example, the space before
            "dog" and the letters "d-o-g" should be treated as a unit; we do not
            want a merge to bridge "cat" and " dog" into a single token.

        Steps:
          1. Split the chunk on special tokens (they are never merged).
          2. Run the GPT-2 regex on each segment to get pre-tokens.
          3. UTF-8 encode each pre-token and convert to a tuple of single bytes.
          4. Count how often each byte-tuple appears.

        Returns: Counter mapping (b'h', b'e', b'l', b'l', b'o') → frequency.
        """
        special_tokens_pattern = "|".join([re.escape(t) for t in special_tokens])

        counts = Counter()
        chunks = re.split(special_tokens_pattern, text_chunk)
        for chunk in chunks:
            for match in BPETokenizer.PRE_TOKEN_PAT.finditer(chunk):
                token_bytes = match.group().encode("utf-8")
                # Convert b'how' → (b'h', b'o', b'w') — each single byte as its own element.
                res = tuple([token_bytes[i : i + 1] for i in range(len(token_bytes))])
                counts[res] += 1

        return counts
