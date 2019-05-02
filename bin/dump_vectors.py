'''
ELMo usage example with pre-computed and cached context independent
token representations

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''
import argparse
import json

import h5py
from tqdm import trange

from bilm import dump_token_embeddings
from bilm.data import UnicodeCharsVocabulary


def convert_vec(vocab_file, options_file, weight_file, outfile, outtype):
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    print("Loading vocabulary..")
    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)

    embeddings = h5py.File(weight_file, 'r')['embedding']

    if outtype == 'vec':
        print("Dumping embeddings in word2vec format")
        with open(outfile, 'w', encoding='utf-8') as fout:
            for k in trange(vocab.size, desc='Printing embeddings'):
                token = vocab.id_to_word(k)
                emb = embeddings[k]
                fout.write("%s %s\n" % (token, " ".join(map(lambda x: "%0.6f" % x, emb))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility script to obtain plain vectors from ELMo embeddings.')
    parser.add_argument("source", choices=["elmo", "hdf5"],
                        help='The source type. '
                             'Choose "elmo" if the source are weights obtained with "dump_weights.py". '
                             'Choose "hdf5" if the source are the plain embeddings in hdf5 format '
                             '(obtainable with this script when outtype is set to "hdf5").')
    parser.add_argument('--weights', help='Location of the weight file.')
    parser.add_argument('--options', help='Location of the options file.')
    parser.add_argument('--vocab', help='Location of the vocabulary file.')
    parser.add_argument('--outfile', help='Output file.')
    parser.add_argument('--outtype', help='Output type. Options: "hdf5", "vec". Default: "vec".', default='vec')

    args = parser.parse_args()

    if args.source == "elmo":
        dump_token_embeddings(
            args.vocab, args.options, args.weights, args.outfile, args.outtype
        )
    elif args.source == "hdf5":
        convert_vec(
            args.vocab, args.options, args.weights, args.outfile, args.outtype
        )
