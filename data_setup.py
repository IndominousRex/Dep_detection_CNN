from torchtext.data.utils import get_tokenizer
import torchtext
from collections import Counter
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import gzip
import json
import random

# NUM_WORKERS = os.cpu_count()

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)

vocabulary = None
tokenizer = None
NUM_WORDS = None


class RSDD(Dataset):
    def __init__(self, posts, labels, max_len, tok=None, vocab=None, mode="random"):
        self.posts = posts
        self.labels = labels
        self.tok = tok
        self.vocab = vocab
        self.mode = mode
        self.max_length = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(
                choose_posts(
                    self.posts[index], self.tok, self.vocab, self.mode, self.max_length
                )
            ),
            torch.tensor(self.labels[index]),
        )


def extract_to_dict(type="training"):
    # storing the file path for the required zip file
    path = f"D:/Downloads/RSDD/RSDD/{type}.gz"
    print(f"\nLoading {type} posts into dictionaries...")

    # initialising empty dictionaries to store posts and labels separately
    labels = {}
    posts = {}

    # loading from file and returning if dumped pickle file exists
    if os.path.exists(f"{type}_labels.pkl") and os.path.exists(f"{type}_posts.pkl"):
        print("\nFound saved data. Loading from file...")
        with open(f"{type}_labels.pkl", "rb") as lb_file:
            labels = pickle.load(lb_file)

        with open(f"{type}_posts.pkl", "rb") as p_file:
            posts = pickle.load(p_file)

        print("\nDone!\n")
        return posts, labels

    # extracting from file if pickle file does not exist
    else:
        # unzipping
        file = gzip.open(path, "rt")
        c = 0

        for i, line in enumerate(file):
            user_index = c

            data = json.loads(line)[0]

            # one-hot representation
            if data["label"] == "depression":
                labels[user_index] = 1
            elif data["label"] == "control":
                labels[user_index] = 0
            else:
                continue

            posts[user_index] = [post for id, post in data["posts"]]
            c += 1
        file.close()

        # storing as a pickle file to be used later
        pickle.dump(posts, open(f"{type}_posts.pkl", "wb"), protocol=-1)
        pickle.dump(labels, open(f"{type}_labels.pkl", "wb"), protocol=-1)

    print("\nDone!")
    return posts, labels


def make_vocab(posts=None):
    print("\nCreating vocabulary...\n")

    # initializing spacy tokenizer
    tokenizer = get_tokenizer("spacy")

    # loading from file if exists
    if os.path.exists("vocab.pkl"):
        print("\nFound saved vocab. Loading from pickle...")
        vocab_counter = pickle.load(open("vocab.pkl", "rb"))

    else:
        print("\nCreating vocab from training data...")
        vocab_counter = Counter()

        # creating list of posts
        uposts = [uposts for post in posts.values() for uposts in post]

        for post in uposts:
            # counting the number of occurences of each token/word in all of the posts
            vocab_counter.update(tokenizer(post))

        # saving the file
        print("\nSaved in vocab.pkl")
        pickle.dump(vocab_counter, open("vocab.pkl", "wb"), protocol=-1)

    ##creating vocabulary as a torchtext vocab object
    unknown_token = "<unk>"
    vocabulary = torchtext.vocab.vocab(
        vocab_counter, min_freq=2, specials=[unknown_token]
    )

    # setting default unidentified token as the unknown token
    vocabulary.set_default_index(vocabulary[unknown_token])

    print(f"There are {len(vocabulary)} words in the vocabulary")

    return tokenizer, vocabulary, len(vocabulary)


def encode(tokenizer, vocab, text):
    # tokenizes the sentence and returns the list of the integer values of the respective tokens
    return [vocab[x] for x in tokenizer(text)]


def pad_text(array, max_len):
    if len(array) >= max_len:
        return array[:max_len]
    else:
        # padding the encoded sentences with zeroes if their length is less than the max length
        return np.pad(
            array, [0, max_len - len(array)], constant_values=[0], mode="constant"
        )


def pad_posts(posts, max_posts, tok, vocab, max_len):
    # encoding text and padding words
    seqs = [encode(tok, vocab, i) for i in posts]
    seqs = [pad_text(i, max_len) for i in seqs]

    # padding posts if number of posts are lower than max posts
    if len(seqs) < max_posts:
        padded = np.pad(seqs, [(0, max_posts - len(seqs)), (0, 0)])
        return padded

    return seqs


def choose_posts(uposts, tok, vocab, mode, max_len):
    # "earliest" mode chooses the earliest 400 posts of the user while random mode choose 1500 posts from the user randomly
    if mode == "earliest":
        max_posts = 400
        chosen = uposts[:max_posts]
        chosen = pad_posts(chosen, max_posts, tok, vocab, max_len)
    else:
        max_posts = 1500
        chosen = pad_posts(uposts, max_posts, tok, vocab, max_len)
        chosen = random.sample(list(chosen), k=max_posts)

    return np.array(chosen)


def get_dataloaders(type, max_length, batch_size, mode):
    global vocabulary, tokenizer

    if vocabulary is None:
        tokenizer, vocabulary, num_words = make_vocab(None)

    # getting data into dictionaries
    posts, labels = extract_to_dict(type)

    # making the dataset
    custom_dataset = RSDD(
        mode=mode,
        posts=posts,
        labels=labels,
        tok=tokenizer,
        vocab=vocabulary,
        max_len=max_length,
    )

    custom_dataloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    if type == "training":
        return custom_dataloader, num_words

    return custom_dataloader
