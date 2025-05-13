from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import numpy as np
import librosa
import pickle
import json
import torch
import torch.nn as nn
from util import MusicTransformer
from generate_audio_from_tokens import generate_wav_from_tokens

WATCH_DIR = "../uploads"
GMM_PATH = "./gmm_model.pkl"
WEIGHTS_PATH = "./music_transformer_weights.pth"

HOP_LENGTH = 512
N_MELS = 128

def load_gmm():
    with open(GMM_PATH, "rb") as f:
        gmm = pickle.load(f)
    return gmm

def number_to_letter(n):
    return chr(ord('a') + n )

def load_file(path):
    y, sr = librosa.load(path, sr=None)
    y = librosa.util.normalize(y)
    return y, sr

def get_onsets(y, sr, hop_length = 512):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_samples=[onsets[0]//2]
    for i in range(len(onsets)-1):
        avg=(onsets[i]+onsets[i+1])//2
        onset_samples.append(avg)
    onset_samples = librosa.frames_to_samples(onset_samples)
    return np.concatenate((onset_samples, [len(y)]))

def get_intervals(onsets):
    out = []
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i+1]
        out.append((start, end))
    return np.array(out)

def get_eigenvalues(y, sr, intervals, hop_length = 512):
    out = []
    for interval in intervals:
        segment = y[interval[0]:interval[1]]
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, hop_length=hop_length, n_fft=512)
        G = mel @ mel.T
        eigvals = np.linalg.eigvals(G)
        out.append(np.real(eigvals))
    return np.array(out)

def subtokenize(file_path):
    gmm = load_gmm()
    y, sr = load_file(file_path)
    onsets = get_onsets(y, sr)
    intervals = get_intervals(onsets)
    eigenvalues = get_eigenvalues(y, sr, intervals)
    sub_tokens = []
    for val in eigenvalues:
        cluster_id = gmm.predict(val.reshape(1, -1))[0]
        sub_token = number_to_letter(cluster_id)
        sub_tokens.append(sub_token)
    return sub_tokens

class TrieNode:
    __slots__ = ("children", "token_id")
    def __init__(self):
        self.children = {}
        self.token_id = None

def build_trie(vocab_list):
    """
    Build a trie where each path is one merge-token (as a sequence of single-char tokens).
    """
    root = TrieNode()
    for idx, merge in enumerate(vocab_list):
        node = root
        for ch in merge:             # each atomic token is one character
            node = node.children.setdefault(ch, TrieNode())
        node.token_id = idx
    return root

def tokenize(tokens, trie, fallback_map):
    """
    Single-pass longest-prefix match against the trie.
    If no multi-char merge matches at i, fall back to single-char token.
    """
    out_ids = []
    n = len(tokens)
    i = 0
    while i < n:
        node = trie
        last_id = None
        last_len = 0
        j = i

        # walk as far as we can
        while j < n and tokens[j] in node.children:
            node = node.children[tokens[j]]
            j += 1
            if node.token_id is not None:
                last_id = node.token_id
                last_len = j - i

        if last_len > 0:
            out_ids.append(last_id)
            i += last_len
        else:
            # single-char fallback
            out_ids.append(fallback_map[tokens[i]])
            i += 1

    return out_ids

def predict(
    model: nn.Module,
    token_ids: str,
    device: torch.device,
    context_size: int = 512,
    num_tokens: int = 50
):
    """
    Autoregressively generate `num_tokens` new tokens following
    the content of `file_path`, using greedy (argmax) decoding.
    """
    # 1) Load & tokenize file
    
    context = token_ids[-context_size:]
    input_ids = torch.tensor([context], dtype=torch.long, device=device)

    model.eval()
    generated_ids = []

    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(input_ids)                 # [1, T, V]
            last_logits = logits[0, -1, :]            # [V]
            next_id = torch.argmax(last_logits).item()# greedy pick
            generated_ids.append(next_id)

            # append and trim to keep at most context_size
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if input_ids.size(1) > context_size:
                input_ids = input_ids[:, -context_size:]

    return generated_ids

def id_to_string(token_ids):
    with open("id_to_token.json", "r") as f:
        D=json.load(f)
    return [D[str(id)] for id in token_ids]

def process_file(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing: {file_path}")
    with open("token_to_id.json", "r") as f:
        token_to_id = json.load(f)
        vocab_list = list(token_to_id.keys())
    sub_tokens = subtokenize(file_path)
    trie = build_trie(vocab_list)
    tokens = tokenize(sub_tokens, trie, token_to_id)
    print("[*] Tokenized sound")
    model = MusicTransformer(
        vocab_size=150,
        d_model=256,
        nhead=2,
        num_layers=6,
        dim_feedforward=1024,
        max_len=512
    )
    model.to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    next_50_tokens = id_to_string(predict(model, tokens, device))
    print("[*] Output received")
    generate_wav_from_tokens(next_50_tokens, file_path[11:-4])
    print("[*] Sound synthesised")


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_file(event.src_path)

if __name__ == "__main__":
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Watching {WATCH_DIR} for new files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()