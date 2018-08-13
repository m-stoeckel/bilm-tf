import argparse
import operator
import os, errno

import numpy as np
import spacy
from tqdm import tqdm
from multiprocessing import cpu_count

from bilm.data import BidirectionalLMDataset
from bilm.training import train, load_vocab


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def pre_process(train_model, train_prefix, vocab_file, heldout_prefix, n_slices=100, min_count=5):
    model_path, model_name = os.path.split(train_model)
    train_path, train_name = os.path.split(train_prefix)
    heldout_path, heldout_name = os.path.split(heldout_prefix)

    for directory in [train_path, heldout_path]:
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    if train_name == "*":
        train_name = model_name

    freq = {}
    curr_buffer = []

    # Source: https://stackoverflow.com/a/9631635
    print("Counting lines..")
    with open(train_model, 'r+', encoding="utf8", errors='ignore') as f_in:
        n_lines = sum(bl.count("\n") for bl in blocks(f_in))

    n_lines_per_slice = n_lines / n_slices

    n_tokens = np.ulonglong(0)
    n_vocab = np.ulonglong(0)

    curr_line = 0
    curr_file_id = 0
    curr_file = open(os.path.join(heldout_path, train_name + ".0"), 'w', encoding='utf8')

    tokenizer = spacy.load('de', disable=['parser', 'tagger', 'ner'])

    print("Reading data..")
    with open(train_model, 'r', encoding="utf8") as f_in:
        for doc in tqdm(tokenizer.pipe(f_in, batch_size=100, n_threads=cpu_count()), total=n_lines):
            tokens = [t.text for t in doc]
            tokenized_line = " ".join(tokens)
            if curr_file_id != int(curr_line / n_lines_per_slice):
                curr_file.writelines(curr_buffer)
                curr_file.close()
                del curr_buffer
                curr_buffer = []
                curr_file_id = int(curr_line / n_lines_per_slice)
                curr_file = open(os.path.join(train_path, train_name + "." + str(curr_file_id)), 'w', encoding='utf8')

            if curr_file_id != 0:
                for token in tokens:
                    if freq.__contains__(token):
                        c = freq[token]
                    else:
                        c = 0
                    freq.update({token: c + 1})
                    n_tokens += 1

            curr_line += 1
            curr_buffer.append(tokenized_line)

    curr_file.writelines(curr_buffer)
    curr_file.close()
    del curr_buffer

    print("Writing vocabulary..")
    with open(vocab_file, 'w', encoding='utf8') as f_out:
        f_out.write("<S>\n")
        f_out.write("</S>\n")
        f_out.write("<UNK>\n")
        for token, count in tqdm(sorted(freq.items(), key=operator.itemgetter(1), reverse=True)):
            if count >= min_count:
                f_out.write(token + '\n')
                n_vocab += 1

    print("Writing heldout..")
    with open(os.path.join(heldout_path, train_name + ".0"), 'r', encoding='utf8') as f_in:
        heldout = []

        curr_heldout_id = 0
        curr_line = 0
        curr_file = open(os.path.join(heldout_path, heldout_name + ".0"), 'w', encoding='utf8')

        n_heldout_lines = sum(bl.count("\n") for bl in blocks(f_in))
        n_lines_per_slice = n_heldout_lines / 50

        for line in f_in:
            if curr_heldout_id != int(curr_line / n_lines_per_slice):
                curr_file.writelines(heldout)
                curr_file.close()
                del heldout
                heldout = []
                curr_heldout_id = int(curr_line / n_lines_per_slice)
                curr_file = open(os.path.join(heldout_path, heldout_name + "." + str(curr_file_id)), 'w',
                                 encoding='utf8')

            heldout.append(line)
            curr_line += 1

        curr_file.writelines(heldout)
        curr_file.close()
        del heldout

    print("Writings stats..")
    with open(train_model + ".stat", 'w', encoding='utf8') as f_out:
        f_out.write("n_tokens:" + str(n_tokens) + "\n")
        f_out.write("n_vocab:" + str(n_vocab) + "\n")

    print("n_lines:" + str(n_lines))
    print("n_tokens:" + str(n_tokens))
    print("n_vocab:" + str(n_vocab))

    del freq

    print("Finished pre-processing.")
    return n_tokens


def main(args):
    if args.pre_process:
        n_train_tokens = pre_process(args.pre_process, args.train_prefix, args.vocab_file, args.heldout_prefix, args.min_count)
    elif args.n_tokens:
        n_train_tokens = args.n_tokens
    elif args.stat:
        with open(args.stat, 'r', encoding='utf8') as f_in:
            args = dict((key, value) for (key, value) in [line.split(":")[:2] for line in f_in.readlines()])
            n_train_tokens = args.get("n_tokens")
    else:
        raise ValueError("Missing token number!")

    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = args.batchsize  # batch size for each GPU
    n_gpus = args.use_gpus

    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 261,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': args.epochs,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    print("Loaded data.")
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')

    parser.add_argument('--pre_process', help='The model to pre-process.')
    parser.add_argument('--heldout_prefix', help='The path and prefix for heldout files.')

    parser.add_argument('--use_gpus', help='The number of gpus to use', type=int, default=2)
    parser.add_argument('--epochs', help='The number of epochs to run', type=int, default=10)
    parser.add_argument('--batchsize', help='The batchsize for each gpu', type=int, default=128)
    parser.add_argument('--min_count', help='The minimal count for a vocabulary item.', type=int, default=5)

    parser.add_argument('--stats', help='Use a .stat file for input data statistics, like token count.')
    parser.add_argument('--n_tokens', help='The number of tokens in the training files')

    args = parser.parse_args()
    main(args)
