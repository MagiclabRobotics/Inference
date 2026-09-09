import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_pi05_returns_state_owner():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    state = np.zeros((2, 3), dtype=np.float32)
    tokens, masks, owner = tokenizer.tokenize(
        "Hello, world!",
        state,
        return_state_token_owner=True,
    )

    assert tokens.shape == (64,)
    assert masks.shape == (64,)
    assert owner.shape == (64,)
    assert np.any(owner == 0)
    assert np.any(owner == 1)
    assert np.all(owner[~masks] == -1)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
