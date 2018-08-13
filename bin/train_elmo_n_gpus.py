import argparse
import operator
import os

import numpy as np
from tqdm import tqdm

from bilm.data import BidirectionalLMDataset
from bilm.training import train, load_vocab


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def pre_process(file_in, path_out, vocab_file, heldout_file, n_slices=5000):
    file_name = file_name = os.path.split(file_in)[1]
#    print(file_name)

    freq = {}
    curr_buffer = []
    heldout = []

    # Source: https://stackoverflow.com/a/9631635
    print("Counting lines..")
    with open(file_in, 'r+', encoding="utf8", errors='ignore') as f_in:
        n_lines = sum(bl.count("\n") for bl in blocks(f_in))

    n_lines_per_slice = n_lines / n_slices

    n_tokens = np.ulonglong(0)
    n_heldout = 0

    curr_line = 0
    curr_file_id = 0
    curr_file = open(os.path.join(path_out.replace("*", ""), file_name + "." + str(curr_file_id)), 'w', encoding='utf8')
#    print(os.path.join(path_out.replace("*", ""), file_name + "." + str(curr_file_id)))

    print("Reading data..")
    with open(file_in, 'r', encoding="utf8") as f_in:
        for line in tqdm(f_in, total=n_lines):
            if curr_file_id != int(curr_line / n_lines_per_slice):
                curr_file.writelines(curr_buffer)
                del curr_buffer
                curr_buffer = []
                curr_file_id = int(curr_line / n_lines_per_slice)
                curr_file = open(os.path.join(path_out.replace("*", ""), file_name + "." + str(curr_file_id)), 'w', encoding='utf8')

            if n_heldout < 1000 and curr_line % 1000 == 0:
                n_heldout += 1
                heldout.append(line)
            else:
                for token in line.split():
                    if freq.__contains__(token):
                        c = freq[token]
                    else:
                        c = 0
                    freq.update({token: c + 1})
                    n_tokens += 1

            curr_line += 1
            curr_buffer.append(line)

    curr_file.writelines(curr_buffer)
    del curr_buffer

    print("Writing vocabulary..")
    with open(vocab_file, 'w', encoding='utf8') as f_out:
        f_out.write("<S>\n")
        f_out.write("</S>\n")
        f_out.write("<UNK>\n")
        for token, count in tqdm(sorted(freq.items(), key=operator.itemgetter(1), reverse=True)):
            if count > 4:
                f_out.write(token + '\n')

    print("Writing heldout..")
    with open(heldout_file, 'w', encoding='utf8') as f_out:
        f_out.writelines(heldout)

    print("n_lines:" + str(n_lines))
    print("n_tokens:" + str(n_tokens))

    del freq
    del heldout
    del curr_buffer

    print("Finished pre-processing.")
    return n_tokens


def main(args):
    if args.pre_process:
        n_train_tokens = pre_process(args.pre_process, args.train_prefix, args.vocab_file, args.heldout)
    else:
        n_train_tokens = args.n_tokens

    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = int(args.use_gpus)

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

        'n_epochs': 100,
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
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--heldout', help='Heldout file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_tokens', help='The number of tokens in the training files')
    parser.add_argument('--pre_process', help='The not pre-processed training file')
    parser.add_argument('--use_gpus', help='The number of gpus to use')

    args = parser.parse_args()
    main(args)
