"""Directly moved here from 4M-21 train_wordpiece_tokenizer.py"""

# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
from collections import defaultdict
from typing import Optional, Union

from tokenizers import AddedToken, Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer


def generate_sentinel_tokens(num=100, start_id=0):
    tokens = [
        AddedToken(content=f"[S_{i}]", single_word=True, normalized=False)
        for i in range(start_id, num + start_id)
    ]
    return tokens


def generate_coord_tokens(bins=1000):
    tokens = []
    coords_str = ["xmin={}", "ymin={}", "xmax={}", "ymax={}"]
    for s in coords_str:
        for i in range(bins):
            tokens.append(
                AddedToken(content=s.format(i), single_word=True, normalized=False)
            )
    return tokens


def generate_object_class_tokens(dataset="coco"):
    with open(os.path.join(os.path.dirname(__file__), "object_classes.json")) as f:
        object_classes = json.load(f)[dataset]

    tokens = [
        AddedToken(content=class_name, single_word=True, normalized=True)
        for class_name in object_classes
    ]
    return tokens


def train_unified_wordpiece_tokenizer(
    files,
    vocab_size,
    sentinel_tokens: list[Union[str, AddedToken]] = None,
    coord_tokens: list[Union[str, AddedToken]] = None,
    object_class_tokens: list[Union[str, AddedToken]] = None,
    unk_token: Union[str, AddedToken] = "[UNK]",
    pad_token: Union[str, AddedToken] = "[PAD]",
    sos_token: Union[str, AddedToken] = "[SOS]",
    eos_token: Union[str, AddedToken] = "[EOS]",
    additional_special_tokens: list[Union[str, AddedToken]] = None,
    min_frequency=0,
    clean_text: bool = True,
    handle_chinese_chars: bool = True,
    strip_accents: bool = True,
    lowercase: bool = True,
    wordpieces_prefix: str = "##",
    show_progress=True,
):
    tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

    tokenizer.normalizer = BertNormalizer(
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

    special_tokens = []
    special_tokens.append(pad_token)
    special_tokens.append(unk_token)
    special_tokens.append(sos_token)
    special_tokens.append(eos_token)

    if sentinel_tokens is not None:
        special_tokens.extend(sentinel_tokens)
    if coord_tokens is not None:
        special_tokens.extend(coord_tokens)
    if object_class_tokens is not None:
        special_tokens.extend(object_class_tokens)
    if additional_special_tokens is not None:
        special_tokens.extend(additional_special_tokens)

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        continuing_subword_prefix=wordpieces_prefix,
        special_tokens=special_tokens,
    )

    if isinstance(files, str):
        files = [files]

    tokenizer.train(files, trainer=trainer)

    return tokenizer


def get_sentinel_to_id_mapping(tokenizer, match_str="[S_"):
    sentinel_tokens = {
        k: v for k, v in tokenizer.get_vocab().items() if k.startswith(match_str)
    }
    # Extract the sentinel token id, the id is of the form "[S_0]", "[S_1]", etc.
    sentinel_to_id = {
        int(k.split("_")[1][:-1]): v
        for k, v in sorted(sentinel_tokens.items(), key=lambda x: x[1])
    }
    return sentinel_to_id


def split_by_sentinel(seq_ids, sentinel_ids):
    splits = defaultdict(list)
    cur_sentinel = None
    for token in seq_ids:
        if token in sentinel_ids:
            cur_sentinel = token
        else:
            splits[cur_sentinel].append(token)

    return splits


def merge_span_masking(input_seq, decoder_seq, sentinel_ids):
    decoder_splits = split_by_sentinel(decoder_seq, sentinel_ids)
    out_seq = []
    for token in input_seq:
        if token in sentinel_ids:
            out_seq.extend(decoder_splits[token])
        else:
            out_seq.append(token)
    return out_seq


def get_args():
    parser = argparse.ArgumentParser(
        "Train unified WordPiece tokenizer", add_help=False
    )
    parser.add_argument(
        "--text_files",
        type=str,
        help="Files to train the tokenizer on, separated by a double dash '--'",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="trained_tokenizer/wordpiece.json",
        help="Path to the saved tokenizer. Can then be loaded using Tokenizer.from_file(path).",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30_000, help="Vocabulary size"
    )
    parser.add_argument(
        "--num_sentinels", type=int, default=0, help="Number of sentinel tokens (set to 0 to disable)"
    )
    parser.add_argument(
        "--coord_bins",
        type=int,
        default=0,
        help="Number of coordinate bins (for detection, set to 0 to disable)",
    )
    parser.add_argument(
        "--object_classes",
        type=str,
        default="none",
        choices=["none", "coco"],
        help="Special tokens for detection instances (e.g., instance class names from the COCO dataset)",
    )
    return parser.parse_args()


def train_tokenizer(args):
    files = args.text_files.split("--")
    # Generate sentinel tokens optionally
    if args.num_sentinels and args.num_sentinels > 0:
        sentinel_tokens = generate_sentinel_tokens(num=args.num_sentinels)
    else:
        sentinel_tokens = None
    # Generate coordinate tokens optionally
    if args.coord_bins and args.coord_bins > 0:
        coord_tokens = generate_coord_tokens(bins=args.coord_bins)
    else:
        coord_tokens = None
    # Generate object class tokens optionally
    if args.object_classes == "none":
        object_class_tokens = None
    else:
        object_class_tokens = generate_object_class_tokens(args.object_classes)

    print(f"Training tokenizer on files: {files}")

    # Train tokenizer
    tokenizer = train_unified_wordpiece_tokenizer(
        files=files,
        vocab_size=args.vocab_size,
        sentinel_tokens=sentinel_tokens,
        coord_tokens=coord_tokens,
        object_class_tokens=object_class_tokens,
    )

    # Create directory of target file if it doesn't exist
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    tokenizer.save(path=args.save_file)

    print(f"Tokenizer saved to: {args.save_file}!")


if __name__ == "__main__":
    args = get_args()
    train_tokenizer(args)
