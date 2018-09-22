
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import numpy as np
import os
import copy
import pdb
import random
from functools import reduce
import tensorflow as tf
import math
import sys
import argparse
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.tensorboard.plugins import projector

import collections
from tempfile import gettempdir
import zipfile

import numpy as np


# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


class Data():
    def __init__(self, H, Q, A):
        self.h = H
        self.q = Q
        self.a = A
        self.datalen = len(self.h)
        self.batch_startidx = 0

    def next_batch(self, batch_size, shuffle=True):
        finishOneEpoch = False

        def shuffle_data():
            perm = np.arange(self.datalen)
            np.random.shuffle(perm)
            self.h = self.h[perm]
            self.q = self.q[perm]
            self.a = self.a[perm]

        if shuffle and self.batch_startidx == 0:
            shuffle_data()

        if self.batch_startidx + batch_size > self.datalen:
            batch_numrest = self.datalen - self.batch_startidx
            rest_H = self.h[self.batch_startidx:]
            rest_Q = self.q[self.batch_startidx:]
            rest_A = self.a[self.batch_startidx:]
            if shuffle:
                shuffle_data()
            num_next = batch_size - batch_numrest
            next_H = self.h[0:num_next]
            next_Q = self.q[0:num_next]
            next_A = self.a[0:num_next]
            batch_h = np.concatenate([rest_H, next_H], axis=0)
            batch_q = np.concatenate([rest_Q, next_Q], axis=0)
            batch_a = np.concatenate([rest_A, next_A], axis=0)
            self.batch_startidx = num_next
            finishOneEpoch = True
        else:
            batch_h = self.h[self.batch_startidx:self.batch_startidx + batch_size]
            batch_q = self.q[self.batch_startidx:self.batch_startidx + batch_size]
            batch_a = self.a[self.batch_startidx:self.batch_startidx + batch_size]
            self.batch_startidx += batch_size
        return batch_h, batch_q, batch_a, finishOneEpoch


class GenData():
    def __init__(self, datadir, memory_size=5, description_size=10, vocab_size=-1, taskid=-1):
        self.fvocab = os.path.join(datadir, 'vocab.out')
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.description_size = description_size
        self.get_data_from_raw(datadir, taskid)

    def get_data_from_raw(self, datadir, taskid=-1):
        train_raw = []
        test_raw = []
        nload = 0
        filenameformat = "qa{}_".format(taskid)
        for file in os.listdir(datadir):
            if taskid == -1 or filenameformat in file and 'train' in file:
                print("file", file)
                train_raw += self.read_source(os.path.join(datadir, file))
            elif taskid == -1 or filenameformat in file and 'test' in file:
                print("file", file)
                test_raw += self.read_source(os.path.join(datadir, file))

        self.statistics(train_raw)
        self.vocab = self.buildDict(train_raw)

        H, Q, A = self.vectorize(train_raw)
        self.train = Data(H, Q, A)

        H, Q, A = self.vectorize(test_raw)
        self.test = Data(H, Q, A)

    def read_source(self, file):
        def parse_line(line):
            line = line.strip()
            if line[-1] == '.' or '?':
                line = line[0:-1]
            line = line.split(' ')
            ID = line[0]
            line = line[1:]
            return ID, line

        history = []
        dataset = []
        with open(file) as f:
            for line in f:
                line = line.lower().strip()
                if len(line.split('\t')) == 3:  # is_question(line):
                    q, a, _ = line.split('\t')
                    _, q = parse_line(q)
                    _history = copy.deepcopy(history)
                    dataset.append((_history, q, [a]))
                else:
                    id, line = parse_line(line)
                    if id == '1':
                        history = []
                    history.append(line)

        return dataset

    def statistics(self, train_raw):
        history_len = [len(h) for h, q, a in train_raw]
        self.max_history_length = max(history_len)
        self.mean_history_length = np.mean(history_len)

        self.max_history_sentence_length = max(
            [len(x) for x in reduce(lambda x, y:x+y, [h for h, q, ans in train_raw])])
        self.max_query_sentence_length = max([len(q) for h, q, a in train_raw])
        print ('===============================================')
        print ('\t\tmax_history_length: {0}\n\
\t\tmean_history_length: {1}\n\
\t\tmax_history_sentence_length: {2}\n\
\t\tmax_query_sentence_length:{3}'.format(
            self.max_history_length,
            self.mean_history_length,
            self.max_history_sentence_length,
            self.max_query_sentence_length))

    def buildDict(self, train_raw):
        def summ(x, y): return x+y
        allwords = reduce(
            summ, [reduce(summ, h) + q + a for h, q, a in train_raw])
        vocab = collections.Counter(allwords)

        vocab_sort = sorted(vocab, key=vocab.get, reverse=True)
        # print vocabulary to file
        with open(self.fvocab, 'w') as f:
            for word in vocab_sort:
                print(f, '\t'.join([word, str(vocab[word])]))
        print ('===============================================')
        print ('written vocabulary to ', self.fvocab)
        self.vocab_size = self.vocab_size if self.vocab_size != - \
            1 else len(vocab_sort) + 2
        vocab = sorted(
            zip(vocab_sort[0: self.vocab_size - 2], range(1, self.vocab_size - 1)))
        filename = maybe_download('text8.zip', 31344016)
        vocabulary = read_data(filename)
        print('Data size is here', len(vocabulary))

        data, count, dictionary, reverse_dictionary = build_dataset(
            vocabulary, len(vocabulary))


        # del vocab  # Hint to reduce memory.

        # print('Most common words (+UNK)', count[:5])
        # print('Sample data', data[:10], [
        #       reverse_dictionary[i] for i in data[:10]])

        # print("DICTIONARY", dictionary)
        # print("REVERSED DICTIONARY", reverse_dictionary)

        batch, labels = generate_batch(
            data, batch_size=8, num_skips=2, skip_window=1)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
                  reverse_dictionary[labels[i, 0]])

        batch_size = 128
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 1  # How many words to consider left and right.
        num_skips = 2  # How many times to reuse an input to generate a label.
        num_sampled = 64  # Number of negative examples to sample.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. These 3 variables are used only for
        # displaying model accuracy, they don't affect calculation.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        graph = tf.Graph()

        with graph.as_default():

            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal(
                            [self.vocab_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=self.vocab_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

        # Step 5: Begin training.
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            # print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
                filename = maybe_download('text8.zip', 31344016)
                vocabulary = read_data(filename)
                print('Data size is here', len(vocabulary))

                data, count, dictionary, reverse_dictionary = build_dataset(
                vocabulary, len(vocabulary))
                batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # print('run_metadata', feed_dict);

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
                for i in xrange(self.vocab_size):
                    f.write(reverse_dictionary[i] + '\n')

            # Save the model for checkpoints.
            saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
            projector.visualize_embeddings(writer, config)

        writer.close()

        # add <unk> and <nil> to vocabulary

        vocab.append(('<nil>', 0))
        vocab.append(('<unk>', self.vocab_size - 1))
        assert self.vocab_size == len(vocab)
        print ('vocabulary size:', self.vocab_size)
        return dict(vocab)

    def vectorize(self, raw):

        def complete_sentence(sent, length):
            """
            complete a sentence to specific "length" with <nil> and turn it into IDs
            """
            sent = [self.vocab.get(w, self.vocab['<unk>']) for w in sent]
            if len(sent) > length:
                sent = sent[0:length]
            else:
                sent += [self.vocab['<nil>']] * (length - len(sent))
            return sent

        H = []
        Q = []
        A = []
        for rawH, rawQ, rawA in raw:
            # deal with history
            idxH = copy.deepcopy(rawH)
            if len(idxH) > self.memory_size:  # only remain the lastest memory_size ones
                idxH = idxH[len(idxH) - self.memory_size:]
            for idx, h in enumerate(idxH):
                idxH[idx] = complete_sentence(h, self.description_size)
            idxH += [[self.vocab['<nil>']] *
                     self.description_size for _ in range(self.memory_size - len(idxH))]

            # deal with question
            idxQ = complete_sentence(rawQ, self.description_size)

            # deal with answer
            idxA = [0] * len(self.vocab)
            idxA[self.vocab.get(rawA[0], self.vocab['<unk>'])] = 1

            H.append(idxH)
            Q.append(idxQ)
            A.append(idxA)

        return np.asarray(H), np.asarray(Q), np.asarray(A)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data



data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(
        maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('')
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


if __name__ == '__main__':
    data = GenData('../memn2n/data/tasks_1-20_v1-2/en',
                   memory_size=15, description_size=10, vocab_size=-1, taskid=1)
    for i in range(10):
        batchh, batchq, batcha, finishOneEpoch = data.train.next_batch(
            1200, shuffle=False)
