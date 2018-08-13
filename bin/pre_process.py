import argparse

from bin.train_elmo_n_gpus import pre_process


def main(args):
    pre_process(args.pre_process, args.train_prefix, args.vocab_file, args.heldout_prefix, args.min_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', help='The model to pre-process.')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--heldout_prefix', help='The path and prefix for heldout files.')
    parser.add_argument('--min_count', help='The minimal count for a vocabulary item.', type=int, default=5)

    args = parser.parse_args()
    main(args)
