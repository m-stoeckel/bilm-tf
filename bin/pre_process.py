import argparse
import errno
import operator
import os
import os.path as path

import numpy as np
from tqdm import tqdm


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


class PreProcessor:
    def __init__(self):
        self.options = {}

    def set_options(self, options):
        self.options = options

    def pre_process(self):
        output_path = path.abspath(self.options["output_path"])
        corpus_file = path.abspath(self.options["corpus"])
        corpus_path, corpus_name = path.split(corpus_file)

        vocab_file = path.join(output_path, self.options.get("vocab_file", path.join(output_path, corpus_name)))
        stat_file = path.join(output_path, self.options.get("stat_file", corpus_name + ".stat"))

        n_slices = self.options.get("n_slices", 100)
        min_count = self.options.get("min_count", {5})

        train_path = path.join(output_path, corpus_name + ".training")
        heldout_path = path.join(output_path, corpus_name + ".heldout")
        train_prefix = path.join(train_path, "training." + corpus_name)
        heldout_part_name = path.join(heldout_path, "training." + corpus_name + ".0")
        heldout_prefix = path.join(heldout_path, "heldout." + corpus_name)

        for directory in [output_path, train_path, heldout_path]:
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        freq = {}
        curr_buffer = []

        # Source: https://stackoverflow.com/a/9631635
        n_lines = self.count_lines(corpus_file)

        n_lines_per_slice = n_lines / n_slices

        n_tokens = np.ulonglong(0)

        curr_line = 0
        curr_file_id = 0
        curr_file = open(heldout_part_name, 'w', encoding='utf8')

        print("Reading data..")
        with open(corpus_file, 'r', encoding="utf8") as f_in:
            for line in tqdm(f_in, total=n_lines):
                tokens = line.split()
                if curr_file_id != int(curr_line / n_lines_per_slice):
                    curr_file.writelines(curr_buffer)
                    curr_file.close()
                    del curr_buffer
                    curr_buffer = []
                    curr_file_id = int(curr_line / n_lines_per_slice)
                    curr_file = open(train_prefix + "." + str(curr_file_id), 'w', encoding='utf8')

                if curr_file_id != 0:
                    n_tokens += self.count_occurrences(freq, tokens)

                curr_line += 1
                curr_buffer.append(line)

        curr_file.writelines(curr_buffer)
        curr_file.close()
        del curr_buffer

        print("Writings stats..")
        with open(stat_file, 'w', encoding='utf8') as f_out:
            f_out.write("n_tokens:" + str(n_tokens) + "\n")

        print("n_lines:" + str(n_lines))
        print("n_tokens:" + str(n_tokens))

        self.write_vocab(freq, min_count, vocab_file)

        del freq

        self.process_heldout(heldout_part_name, heldout_prefix)

        print("Finished pre-processing.")
        return n_tokens

    def gen_heldout(self):
        self.process_heldout(self.options["train_name"], self.options["heldout_prefix"])

    def process_heldout(self, train_name, heldout_prefix):
        print("Writing heldout..")
        n_heldout_lines = self.count_lines(train_name)
        with open(train_name, 'r', encoding='utf8') as f_in:
            heldout = []

            curr_heldout_id = 0
            curr_line = 0

            curr_file = open(heldout_prefix + ".0", 'w', encoding='utf8')

            n_lines_per_slice = n_heldout_lines / 50

            for line in tqdm(f_in, total=n_heldout_lines):
                if curr_heldout_id != int(curr_line / n_lines_per_slice):
                    curr_file.writelines(heldout)
                    curr_file.close()
                    del heldout
                    heldout = []
                    curr_heldout_id = int(curr_line / n_lines_per_slice)
                    curr_file = open(heldout_prefix + "." + str(curr_heldout_id), 'w', encoding='utf8')

                heldout.append(line)
                curr_line += 1

            curr_file.writelines(heldout)
            curr_file.close()
            del heldout

    def gen_vocab(self):
        corpus_file = path.abspath(self.options["corpus"])
        output_path = path.abspath(self.options["output_path"])
        vocab_prefix = self.options.get("vocab_file", path.join(output_path, path.split(corpus_file)[1]))
        min_count = self.options.get("min_count", {5})
        if min_count is None or min_count == {}:
            min_count = {5}

        n_lines = self.count_lines(corpus_file)
        n_token = np.ulonglong(0)

        freq = {}

        print("Reading data..")
        with open(corpus_file, 'r', encoding="utf8") as f_in:
            for line in tqdm(f_in, total=n_lines):
                n_token += self.count_occurrences(freq, line.split())

        print("Writing vocab..")
        self.write_vocab(freq, min_count, vocab_prefix)

        print("Wrote vocabs for min_count=" + str(min_count) + " from " + str(n_token) + " tokens.")

    @staticmethod
    def write_vocab(freq, min_count, vocab_file):
        if min_count.__contains__(None):
            min_count.remove(None)
        for min in min_count:
            print("Writing vocabulary with min count " + str(min) + "..")
            n_vocab = np.ulonglong(0)
            with open(vocab_file + "." + str(min) + ".vocab", 'w', encoding='utf8') as f_out:
                f_out.write("<S>\n")
                f_out.write("</S>\n")
                f_out.write("<UNK>\n")
                for token, count in tqdm(sorted(freq.items(), key=operator.itemgetter(1), reverse=True)):
                    if count >= min:
                        f_out.write(token + '\n')
                        n_vocab += 1

    @staticmethod
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

    @staticmethod
    def count_lines(corpus_file):
        """ Source: https://stackoverflow.com/a/9631635 """
        print("Counting lines..")
        with open(corpus_file, 'r', encoding="utf8", errors='ignore') as f_in:
            n_lines = sum(bl.count("\n") for bl in blocks(f_in))
        return n_lines


def main(args):
    pre_processor = PreProcessor()
    opt = {
        "output_path": args.output_path,
        "corpus": args.corpus,
        "vocab_file": args.vocab_file,
        "stat_file": args.stat_file,
        "n_slices": args.n_slices,
        "min_count": args.min_count,
        "train_name": args.train_name,
        "heldout_prefix": args.heldout_prefix
    }
    opt = {k: v for k, v in opt.items() if v is not None}
    pre_processor.set_options(opt)
    if args.pre_process:
        pre_processor.pre_process()
    elif args.gen_vocab:
        pre_processor.gen_vocab()
    elif args.gen_heldout:
        pre_processor.gen_heldout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', help='Pre-process flag. Uses -o, -c, -v and -s.', action='store_true')
    parser.add_argument('--gen_vocab', help='Vocabulary generation only flag. Uses -o, -c, -v and -s.', action='store_true')

    parser.add_argument('-o', '--output_path', help='Output path. Subdirectories for training and heldout and other output files will be created here.')
    parser.add_argument('-c', '--corpus', help='The corpus to pre-process.')
    parser.add_argument('-v', '--vocab_file', help='Vocabulary file name prefix.')
    parser.add_argument('-s', '--stat_file', help='Corpus statistics file name.')

    parser.add_argument('--min_count', help='A list of minimal counts, create a vocab for each. The number is appended to each file name', type=int, nargs='+', default={5})
    parser.add_argument('--n_slices', help='Number of slices', type=int, default=100)

    parser.add_argument('--gen_heldout', help='Heldout processing only flag. Uses -t and -g.', action='store_true')
    parser.add_argument('-t', '--train_name', help='Name of the training slice to use as a heldout source')
    parser.add_argument('-g', '--heldout_prefix', help='Prefix for heldout files, in form of "path/to/heldout/heldout_file".')

    args = parser.parse_args()
    main(args)
