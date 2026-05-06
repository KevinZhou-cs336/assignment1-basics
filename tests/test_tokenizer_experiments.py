import pathlib
import time

import numpy as np

from cs336_basics.bpe.bpe import BPETokenizer

DATA_PATH = (pathlib.Path(__file__).resolve().parent) / ".." / "data"

TINY_STORIES_VOCAB   = DATA_PATH / "tinyStories_Vocab.file"
TINY_STORIES_MERGES  = DATA_PATH / "tinyStories_Merges.file"
OWT_VOCAB            = DATA_PATH / "openwebtext_Vocab.file"
OWT_MERGES           = DATA_PATH / "openwebtext_Merges.file"

SPECIAL_TOKENS = ["<|endoftext|>"]
DOC_SEP = "<|endoftext|>"


def _sample_documents(filepath: pathlib.Path, n: int = 10) -> list[str]:
    """Read the file and return the first n non-empty documents split on DOC_SEP."""
    with open(filepath, "r", errors="replace") as f:
        raw = f.read(50 * 1024 * 1024)  # read up to 50 MB to find 10 docs
    docs = [d.strip() for d in raw.split(DOC_SEP) if d.strip()]
    return docs[:n]


def _compression_ratio(tokenizer: BPETokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


# 2.7 (a) ─ Compression ratio on home dataset
def test_tokenizer_compression_ratio():
    """
    2.7(a): Sample 10 docs from each dataset, encode with the matching tokenizer,
    report compression ratio (bytes/token).
    """
    ts_tokenizer  = BPETokenizer.from_file(TINY_STORIES_VOCAB, TINY_STORIES_MERGES, SPECIAL_TOKENS)
    owt_tokenizer = BPETokenizer.from_file(OWT_VOCAB, OWT_MERGES, SPECIAL_TOKENS)

    ts_docs  = _sample_documents(DATA_PATH / "TinyStoriesV2-GPT4-train.txt")
    owt_docs = _sample_documents(DATA_PATH / "owt_train.txt")

    ts_ratio  = _compression_ratio(ts_tokenizer,  ts_docs)
    owt_ratio = _compression_ratio(owt_tokenizer, owt_docs)

    # Results (sampled 10 documents per dataset):
    #   TinyStories tokenizer (10K vocab) on TinyStories docs  -> 4.150 bytes/token
    #   OpenWebText tokenizer (32K vocab) on OpenWebText docs  -> 4.691 bytes/token
    # The OWT tokenizer achieves a higher compression ratio because its larger vocabulary
    # captures more diverse, longer subword units from web text.
    print(f"\n[2.7a] TinyStories tokenizer on TinyStories docs  → compression ratio = {ts_ratio:.3f} bytes/token")
    print(f"[2.7a] OpenWebText tokenizer on OpenWebText docs  → compression ratio = {owt_ratio:.3f} bytes/token")


# 2.7 (b) ─ Cross-domain: TinyStories tokenizer on OpenWebText
def test_tokenizer_on_different_data_set():
    """
    2.7(b): Tokenize OpenWebText sample with the TinyStories tokenizer.
    Compare compression ratio vs the native OWT tokenizer.
    """
    ts_tokenizer  = BPETokenizer.from_file(TINY_STORIES_VOCAB, TINY_STORIES_MERGES, SPECIAL_TOKENS)
    owt_tokenizer = BPETokenizer.from_file(OWT_VOCAB, OWT_MERGES, SPECIAL_TOKENS)

    owt_docs = _sample_documents(DATA_PATH / "owt_train.txt")

    ts_ratio  = _compression_ratio(ts_tokenizer,  owt_docs)
    owt_ratio = _compression_ratio(owt_tokenizer, owt_docs)

    # Results (same 10 OWT documents, two different tokenizers):
    #   TinyStories tokenizer (10K vocab) on OpenWebText docs  -> 3.189 bytes/token
    #   OpenWebText tokenizer (32K vocab) on OpenWebText docs  -> 4.691 bytes/token
    #   Compression drop: ~32% worse when using the cross-domain tokenizer.
    # The TinyStories vocabulary is optimised for simple children's stories and lacks the
    # broader subword coverage needed for web text, so it splits OWT words into more and
    # shorter tokens, degrading compression significantly.
    print(f"\n[2.7b] TinyStories tokenizer on OpenWebText docs  → compression ratio = {ts_ratio:.3f} bytes/token")
    print(f"[2.7b] OpenWebText tokenizer on OpenWebText docs  → compression ratio = {owt_ratio:.3f} bytes/token")
    print(f"[2.7b] Ratio drop: {(1 - ts_ratio / owt_ratio) * 100:.1f}% fewer bytes covered per token with cross-domain tokenizer")


# 2.7 (c) ─ Throughput estimate
def test_tokenizer_throughput():
    """
    2.7(c): Measure encode throughput (bytes/sec) and estimate time to tokenize The Pile (825 GB).
    """
    ts_tokenizer = BPETokenizer.from_file(TINY_STORIES_VOCAB, TINY_STORIES_MERGES, SPECIAL_TOKENS)

    SAMPLE_BYTES = 5 * 1024 * 1024  # 5 MB sample for a quick but representative benchmark
    with open(DATA_PATH / "TinyStoriesV2-GPT4-train.txt", "r", errors="replace") as f:
        sample_text = f.read(SAMPLE_BYTES)

    actual_bytes = len(sample_text.encode("utf-8"))
    t0 = time.perf_counter()
    ts_tokenizer.encode(sample_text)
    elapsed = time.perf_counter() - t0

    throughput_mbs = actual_bytes / elapsed / (1024 ** 2)
    pile_gb = 825
    pile_bytes = pile_gb * (1024 ** 3)
    estimated_seconds = pile_bytes / (actual_bytes / elapsed)
    estimated_hours = estimated_seconds / 3600

    # Results (measured on a 5 MB TinyStories sample, Apple M5 Pro):
    #   Throughput = 1.52 MB/s
    #   Estimated time to tokenize The Pile (825 GB) = ~154 hours (~6.4 days)
    # The bottleneck is the pure-Python BPE merge loop; a Rust-based tokenizer
    # (e.g., tiktoken) achieves 100-200x higher throughput on the same hardware.
    print(f"\n[2.7c] Encoded {actual_bytes / 1024 / 1024:.1f} MB in {elapsed:.2f}s")
    print(f"[2.7c] Throughput = {throughput_mbs:.2f} MB/s")
    print(f"[2.7c] Estimated time to tokenize The Pile (825 GB) = {estimated_hours:.1f} hours")


# 2.7 (d) ─ Encode full datasets to numpy uint16
def test_encode_datasets_to_numpy():
    """
    2.7(d): Encode the TinyStories and OpenWebText train/valid splits into
    numpy uint16 arrays and save to disk.
    """
    ts_tokenizer  = BPETokenizer.from_file(TINY_STORIES_VOCAB, TINY_STORIES_MERGES, SPECIAL_TOKENS)
    owt_tokenizer = BPETokenizer.from_file(OWT_VOCAB, OWT_MERGES, SPECIAL_TOKENS)

    datasets = [
        (ts_tokenizer,  DATA_PATH / "TinyStoriesV2-GPT4-train.txt", DATA_PATH / "tinyStories_train.npy"),
        (ts_tokenizer,  DATA_PATH / "TinyStoriesV2-GPT4-valid.txt", DATA_PATH / "tinyStories_valid.npy"),
        (owt_tokenizer, DATA_PATH / "owt_train.txt",                DATA_PATH / "owt_train.npy"),
        (owt_tokenizer, DATA_PATH / "owt_valid.txt",                DATA_PATH / "owt_valid.npy"),
    ]

    for tokenizer, src_path, dst_path in datasets:
        print(f"\n[2.7d] Encoding {src_path.name} → {dst_path.name} ...")
        t0 = time.perf_counter()
        all_ids = []
        with open(src_path, "r", errors="replace") as f:
            for ids in tokenizer.encode_iterable(f):
                all_ids.append(ids)
        arr = np.array(all_ids, dtype=np.uint16)
        np.save(dst_path, arr)
        elapsed = time.perf_counter() - t0
        print(f"[2.7d]   tokens = {len(arr):,}  |  saved to {dst_path.name}  |  {elapsed:.1f}s")
