# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.\

"""Input pipeline for the imdb dataset."""

from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE

DATASET_PATHS = './data/new_aan_pairs'

SHUFFLE_BUFFER_SIZE = 2048


def get_tsv_dataset(file_path, batch_size):
  """Preprocess dataset."""
  tf.logging.info(file_path)
  col_defaults = [tf.float32, tf.string, tf.string, tf.string, tf.string]
  col_names = ['label', 'id1', 'id2', 'text1', 'text2']
  ds = tf.data.experimental.make_csv_dataset([file_path],
                                             batch_size,
                                             column_names=col_names,
                                             column_defaults=col_defaults,
                                             use_quote_delim=False,
                                             field_delim='\t',
                                             header=False,
                                             shuffle=True,
                                             shuffle_buffer_size=10000,
                                             num_epochs=1)
  ds = ds.unbatch()
  return ds


def get_dataset(batch_size):
  """Get dataset from matching datasets converts into src/tgt pairs."""
  train_fps = DATASET_PATHS + '.train.tsv'
  valid_fps = DATASET_PATHS + '.eval.tsv'
  test_fps = DATASET_PATHS + '.test.tsv'
  train = get_tsv_dataset(train_fps, batch_size)
  valid = get_tsv_dataset(valid_fps, batch_size)
  test = get_tsv_dataset(test_fps, batch_size)

  logging.info('Data sample: %s', next(iter(tfds.as_numpy(test.skip(4)))))

  def adapt_example(example):
    return {
        'Source1': example['text1'],
        'Source2': example['text2'],
        'Target': example['label']
    }

  train = train.map(adapt_example)
  valid = valid.map(adapt_example)
  test = test.map(adapt_example)

  return train, valid, test


def get_matching_datasets(n_devices,
                          task_name,
                          data_dir=None,
                          batch_size=256,
                          fixed_vocab=None,
                          max_length=512,
                          tokenizer='subword',
                          vocab_file_path=None):
  """Get text matching classification datasets."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  del data_dir  
  del task_name 

  train_dataset, val_dataset, test_dataset = get_dataset(batch_size)

  tf.logging.info('Finished getting dataset.')

  '''

   byte/character level setup

  '''

  if tokenizer == 'char':
    logging.info('Using char-level/byte dataset..')

    encoder = tfds.features.text.ByteTextEncoder(additional_tokens=None)


  def tf_encode(x):
    result = tf.py_function(
        lambda s: tf.constant(encoder.encode(s.numpy()[:10000])), [
            x,
        ], tf.int32)
    result.set_shape([None])
    return result


  def tokenize(d):
    return ( tf_encode(d['Source1'])[:max_length] ,  tf_encode(d['Source2'])[:max_length] ,   tf.cast(d['Target'], tf.int32)    )



  train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)


  max_shape = ([max_length],[max_length], [])


  train_dataset = train_dataset.shuffle(
      buffer_size=SHUFFLE_BUFFER_SIZE,
      reshuffle_each_iteration=True).padded_batch(
          batch_size, padded_shapes=max_shape, drop_remainder=True)
  val_dataset = val_dataset.padded_batch(
      batch_size, padded_shapes=max_shape, drop_remainder=True)
  test_dataset = test_dataset.padded_batch(
      batch_size, padded_shapes=max_shape, drop_remainder=True)


  def mconcat(d1,d2,d3):

    return  ( tf.concat([d1,d2],axis=-1) ,   d3    )

  train_dataset = train_dataset.map(mconcat, num_parallel_calls=AUTOTUNE)
  val_dataset = val_dataset.map(mconcat, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(mconcat, num_parallel_calls=AUTOTUNE)



  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return train_dataset, val_dataset, test_dataset, encoder
