import argparse
import errno
import operator
import os

import numpy as np
from tqdm import tqdm


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def pre_process(train_corpus, train_prefix, vocab_file, heldout_prefix, n_slices=100, min_count=5):
    corpus_path, corpus_path = os.path.split(train_corpus)
    train_path, train_name = os.path.split(train_prefix)
    heldout_path, heldout_name = os.path.split(heldout_prefix)

    for directory in [train_path, heldout_path]:
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    if train_name == "*":
        train_name = corpus_path

    freq = {}
    curr_buffer = []

    # Source: https://stackoverflow.com/a/9631635
    n_lines = count_lines(train_corpus)

    n_lines_per_slice = n_lines / n_slices

    n_tokens = np.ulonglong(0)
    n_vocab = np.ulonglong(0)

    curr_line = 0
    curr_file_id = 0
    curr_file = open(os.path.join(heldout_path, train_name + ".0"), 'w', encoding='utf8')

    print("Reading data..")
    with open(train_corpus, 'r', encoding="utf8") as f_in:
        for line in tqdm(f_in, total=n_lines):
            tokens = line.split()
            if curr_file_id != int(curr_line / n_lines_per_slice):
                curr_file.writelines(curr_buffer)
                curr_file.close()
                del curr_buffer
                curr_buffer = []
                curr_file_id = int(curr_line / n_lines_per_slice)
                curr_file = open(os.path.join(train_path, train_name + "." + str(curr_file_id)), 'w', encoding='utf8')

            if curr_file_id != 0:
                n_tokens += count_occurrences(freq, tokens)

            curr_line += 1
            curr_buffer.append(line)

    curr_file.writelines(curr_buffer)
    curr_file.close()
    del curr_buffer

    n_vocab = write_vocab(freq, min_count, vocab_file)

    gen_heldout(heldout_path, heldout_name, train_name)

    print("Writings stats..")
    with open(train_corpus + ".stat", 'w', encoding='utf8') as f_out:
        f_out.write("n_tokens:" + str(n_tokens) + "\n")
        f_out.write("n_vocab:" + str(n_vocab) + "\n")

    print("n_lines:" + str(n_lines))
    print("n_tokens:" + str(n_tokens))
    print("n_vocab:" + str(n_vocab))

    del freq

    print("Finished pre-processing.")
    return n_tokens


def gen_heldout(heldout_path, heldout_name, train_name):
    print("Writing heldout..")
    n_heldout_lines = count_lines(os.path.join(heldout_path, train_name + ".0"))
    with open(os.path.join(heldout_path, train_name + ".0"), 'r', encoding='utf8') as f_in:
        heldout = []

        curr_heldout_id = 0
        curr_line = 0

        print("Writing " + os.path.join(heldout_path, heldout_name + ".0"))
        curr_file = open(os.path.join(heldout_path, heldout_name + ".0"), 'w', encoding='utf8')

        n_lines_per_slice = n_heldout_lines / 50

        for line in tqdm(f_in, total=n_heldout_lines):
            if curr_heldout_id != int(curr_line / n_lines_per_slice):
                curr_file.writelines(heldout)
                curr_file.close()
                del heldout
                heldout = []
                curr_heldout_id = int(curr_line / n_lines_per_slice)
                curr_file = open(os.path.join(heldout_path, heldout_name + "." + str(curr_heldout_id)), 'w',
                                 encoding='utf8')

            heldout.append(line)
            curr_line += 1

        curr_file.writelines(heldout)
        curr_file.close()
        del heldout


def write_vocab(freq, min_count, vocab_file):
    print("Writing vocabulary..")
    n_vocab = np.ulonglong(0)
    with open(vocab_file, 'w', encoding='utf8') as f_out:
        f_out.write("<S>\n")
        f_out.write("</S>\n")
        f_out.write("<UNK>\n")
        for token, count in tqdm(sorted(freq.items(), key=operator.itemgetter(1), reverse=True)):
            if count >= min_count:
                f_out.write(token + '\n')
                n_vocab += 1
    return n_vocab


def count_occurrences(freq, tokens):
    count = 0
    for token in tokens:
        if freq.__contains__(token):
            c = freq[token]
        else:
            c = 0
        freq.update({token: c + 1})
        count += 1
    return count


def gen_vocab(corpus_file, vocab_file, min_count=5):
    # Source: https://stackoverflow.com/a/9631635
    n_lines = count_lines(corpus_file)

    freq = {}

    print("Reading data..")
    with open(corpus_file, 'r', encoding="utf8") as f_in:
        for line in tqdm(f_in, total=n_lines):
            count_occurrences(freq, line.split())

    print("Writing vocab..")
    write_vocab(freq, min_count, vocab_file)


def count_lines(corpus_file):
    print("Counting lines..")
    with open(corpus_file, 'r', encoding="utf8", errors='ignore') as f_in:
        n_lines = sum(bl.count("\n") for bl in blocks(f_in))
    return n_lines


def main(args):
    if args.pre_process:
        pre_process(args.pre_process, args.train_prefix, args.vocab_file, args.heldout_prefix, args.min_count)
    elif args.gen_vocab:
        gen_vocab(args.gen_vocab, args.vocab_file, args.min_count)
    elif args.gen_heldout:
        gen_heldout(os.path.split(args.heldout_prefix)[0], os.path.split(args.heldout_prefix)[1],
                    os.path.split(args.train_prefix)[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', help='The corpus to pre-process.')
    parser.add_argument('--gen_vocab', help='Only generate the vocabulary of this corpus, no training data.')
    parser.add_argument('--gen_heldout', help='Only generate the heldout of a given training slice.')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--heldout_prefix', help='The path and prefix for heldout files.')
    parser.add_argument('--min_count', help='The minimal count for a vocabulary item.', type=int, default=5)

    args = parser.parse_args()
    main(args)
