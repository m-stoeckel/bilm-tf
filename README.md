<b>Pre-processing:</b>
Process the input corpus for ELMo training.
Either call with --pre_process or --gen_vocab.
<pre>
usage: pre_process.py [-h] [--pre_process PRE_PROCESS] [--gen_vocab GEN_VOCAB]
                      [--train_prefix TRAIN_PREFIX] [--vocab_file VOCAB_FILE]
                      [--heldout_prefix HELDOUT_PREFIX]
                      [--min_count MIN_COUNT]
optional arguments:
  -h, --help            show this help message and exit
  --pre_process PRE_PROCESS
                        The corpus to pre-process.
  --gen_vocab GEN_VOCAB
                        Only generate the vocabulary of this corpus, no
                        training data.
  --train_prefix TRAIN_PREFIX
                        Prefix for train files
  --vocab_file VOCAB_FILE
                        Vocabulary file
  --heldout_prefix HELDOUT_PREFIX
                        The path and prefix for heldout files.
  --min_count MIN_COUNT
                        The minimal count for a vocabulary item.
Example calls:
python3 bin/pre-process.py --pre_process /home/data/corpora/corpus_a \
            --train_prefix '/home/data/pre/corpus_a_training/train.corpus_a*' \
            --heldout_prefix '/home/data/pre/corpus_a_heldout/heldout.corpus_a*' \
            --vocab_file /home/data/pre/corpus_a.100.vocab \
            --min_count 100
python3 bin/pre-process.py --gen_vocab /home/data/corpora/corpus_a \
            --train_prefix '/home/data/pre/corpus_a_training/train.corpus_a*' \
            --vocab_file /home/data/pre/corpus_a.50.vocab \
            --min_count 50
            
--pre_process: the corpus which to pre-process.
The corpus is split into 100 parts, which are saved with the train_prefix.
A heldout portion (1%) is saved in a different directory and there split into 50 parts.
--train_prefix: The file prefix for training files.
'.$slice_number'  is appended to this name. 
--heldout_prefix: The file prefix for heldout files.
'.$heldout_slice_number' is appended to this name.
The first training slice is also saved into the directory given by this prefix. 
--vocab_file: The file in which the vocabulary is stored. 
--min_count: The minimal count of a token to make it into the vocabulary.
</pre> 
<p/><b>Training:</b>
Train the biLM for ELMo embeddings.
Call with --pre_process to process a corpus first, then train on the generated data.
Call without if data was already pre-processed.
<pre>
usage: train_elmo_n_gpus.py [-h] [--train_prefix TRAIN_PREFIX]
                            [--save_dir SAVE_DIR] [--vocab_file VOCAB_FILE]
                            [--n_tokens N_TOKENS] [--stats STATS]
                            [--use_gpus USE_GPUS] [--epochs EPOCHS]
                            [--batchsize BATCHSIZE]
                            [--pre_process PRE_PROCESS]
                            [--heldout_prefix HELDOUT_PREFIX]
                            [--min_count MIN_COUNT]
optional arguments:
  -h, --help            show this help message and exit
  --train_prefix TRAIN_PREFIX
                        Prefix for train files
  --save_dir SAVE_DIR   Location of checkpoint files
  --vocab_file VOCAB_FILE
                        Vocabulary file
  --n_tokens N_TOKENS   The number of tokens in the training files
  --stats STATS         Use a .stat file for input data statistics, like token
                        count.
  --use_gpus USE_GPUS   The number of gpus to use
  --epochs EPOCHS       The number of epochs to run
  --batchsize BATCHSIZE
                        The batchsize for each gpu
  --pre_process PRE_PROCESS
                        The corpus to pre-process.
  --heldout_prefix HELDOUT_PREFIX
                        The path and prefix for heldout files.
  --min_count MIN_COUNT
                        The minimal count for a vocabulary item.
Example calls:
python3 bin/train_elmo_n_gpus.py \
 --train_prefix '/home/public/stoeckel/data/Leipzig40MT2010_raw_training/train.Leipzig40MT2010_raw*' \
 --vocab_file /home/public/stoeckel/data/Leipzig40MT2010_lowered.100.vocab 
 --save_dir /home/public/stoeckel/models/biLM/Leipzig40MT2010_raw/
 --n_tokens 1093717542 \
</pre>