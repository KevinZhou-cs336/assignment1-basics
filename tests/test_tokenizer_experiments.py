import pathlib

from cs336_basics.bpe import BPETokenizer

DATA_PATH = (pathlib.Path(__file__).resolve().parent) / ".." / "data"


def test_tokenizer_compression_ratio():
    SAMPLE_COUNT = 10
    TINY_STORIES_DATA = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    OPENWEBTEXT_DATA = DATA_PATH / "owt_train.txt"

    TINY_STORIES_VOCAB = DATA_PATH / "tinyStories_Vocab.file"
    TINY_STORIES_MERGES = DATA_PATH / "tinyStories_Merges.file"

    tiny_stories_tokenizer = BPETokenizer.from_file(
        TINY_STORIES_VOCAB, TINY_STORIES_MERGES, ["<|endoftext|>"]
    )


def test_tokenizer_on_different_data_set():
    """
    Use TinyStories tokenizer on openweb sample
    """
    OPENWEBTEXT_DATA = DATA_PATH / "owt_train.txt"

    TINY_STORIES_VOCAB = DATA_PATH / "tinyStories_Vocab.file"
    TINY_STORIES_MERGES = DATA_PATH / "tinyStories_Merges.file"

    tiny_stories_tokenizer = BPETokenizer.from_file(
        TINY_STORIES_VOCAB, TINY_STORIES_MERGES, ["<|endoftext|>"]
    )

    text = None
    with open(OPENWEBTEXT_DATA, "r") as f:
        text = f.read()

    text_bytes_count = len(text.encode("utf-8"))
    print(f"Text bytes count = {text_bytes_count}")
    encoded_text = tiny_stories_tokenizer.encode(text)
    print(f"Compression ratio = {text_bytes_count / len(encoded_text)}")
