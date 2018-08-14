import argparse
from bilm.data import BidirectionalLMDataset
from bilm.training import train, load_vocab
from bin.pre_process import pre_process


def main(args):
    if args.n_tokens:
        n_train_tokens = args.n_tokens
    elif args.stats:
        with open(args.stats, 'r', encoding='utf8') as f_in:
            d = dict((key, value) for (key, value) in [line.split(":")[:2] for line in f_in.readlines()])
            n_train_tokens = d.get("n_tokens")
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
    parser.add_argument('--n_tokens', help='The number of tokens in the training files', type=int)
    parser.add_argument('--stats', help='Use a .stat file for input data statistics, like token count.')

    parser.add_argument('--use_gpus', help='The number of gpus to use', type=int, default=2)
    parser.add_argument('--epochs', help='The number of epochs to run', type=int, default=10)
    parser.add_argument('--batchsize', help='The batchsize for each gpu', type=int, default=128)

    args = parser.parse_args()
    main(args)
