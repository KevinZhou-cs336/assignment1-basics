import pathlib
import pickle
import time
import tracemalloc

from tests.adapters import run_train_bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent) / ".." / "data"


# BPE Training on TinyStories
def test_train_bpe_tinystories():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 30 mins.
    """
    # start mery tracing
    tracemalloc.start()

    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    with open(DATA_PATH / "tinyStories_Vocab.file", "wb") as f:
        pickle.dump(vocab, f)

    with open(DATA_PATH / "tinyStories_Merges.file", "wb") as f:
        pickle.dump(merges, f)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    longest_token = max(vocab.values(), key=len).decode('utf-8')
    print(f"Longest token is {longest_token}")
    print(f"Max memory usage is {peak_mem / 1024 /1024} MB")
    print(f"Spent {end_time-start_time} seconds for training the data set")
    assert end_time - start_time < 1800

# BPE Training on OpenWebText
def test_train_bpe_openwebtext():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 12 * 60 mins.
    """
    # start mery tracing
    tracemalloc.start()

    input_path = DATA_PATH / "owt_train.txt"
    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    with open(DATA_PATH / "openwebtext_Vocab.file", "wb") as f:
        pickle.dump(vocab, f)

    with open(DATA_PATH / "openwebtext_Merges.file", "wb") as f:
        pickle.dump(merges, f)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    longest_token = max(vocab.values(), key=len).decode('utf-8')
    print(f"Longest token is {longest_token}")
    print(f"Max memory usage is {peak_mem / 1024 /1024} MB")
    print(f"Spent {end_time-start_time} seconds for training the data set")
    assert end_time - start_time < 1800
