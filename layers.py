###############################################################################
# Copyright (c) 2018-2019, Tinghui Wang
# License: BSD 3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


import tensorflow as tf
import tensorflow.python.keras as keras


class EmbeddingNCELayer(keras.layers.Layer):
    """Sensor Embedding layer with NCE loss

    This `EmbeddingNCELayer` implements a trainable vector embedding layer
    which is compatible with Keras models. The class shares similar arguments
    and attributes as the `tf.keras.layers.Embedding` layer provided by
    tensorflow.

    Args:
      vocab_size: int > 0. Vocabulary size.
      output_dim: int > 0. Dimension of the dense embedding.
      nce_num_sampled: int > 0. The number of negative samples used in the NCE
        loss function.
      symmetric_embeddings: True, if the weights of generative network shares the
        same value as the embedding matrix.
      embeddings_initializer: Initializer for the `embeddings` matrix.
      embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
      embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
      bias_initializer: Initializer for the `bias` matrix of the generative
        network.
      bias_regularizer: Regularizer function applied to the `bias` matrix of the
        generative network.
      bias_constraint: Constraint function applied to the `bias` matrix of the
        generative network.
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).
    """

    def __init__(self,
                 vocab_size,
                 embeddings_size,
                 nce_num_sampled=None,
                 symmetric_embeddings=False,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(EmbeddingNCELayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        if nce_num_sampled is None:
            self.nce_num_sampled = int(0.1 * vocab_size)
        else:
            self.nce_num_sampled = nce_num_sampled
        self.symmetric_embeddings = symmetric_embeddings
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraints = keras.constraints.get(embeddings_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraints = keras.constraints.get(bias_constraint)

        self.embeddings = None
        self.nce_bias = None
        self.nce_weights = None

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=[self.vocab_size, self.embeddings_size],
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraints,
            trainable=True,
            dtype=tf.float32
        )
        self.nce_bias = self.add_weight(
            shape=(self.vocab_size,),
            initializer=self.bias_initializer,
            name="nce_bias",
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraints,
            trainable=True,
            dtype=tf.float32
        )
        if self.symmetric_embeddings:
            self.nce_weights = self.embeddings
        else:
            self.nce_weights = self.add_weight(
                shape=(self.vocab_size, self.embeddings_size),
                initializer=self.embeddings_initializer,
                name='output_embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraints,
                dtype=tf.float32
            )
        self.build = True

    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            if len(inputs) == 1:
                source = inputs[0]
                labels = None
            else:
                source = inputs[0]
                labels = inputs[1]
        else:
            source = inputs
            labels = None
        source = tf.reshape(source, shape=[-1])
        # inputs has to composed of two settings`
        input_embeddings = tf.nn.embedding_lookup(
            self.embeddings, source
        )
        if labels is not None:
            # nce loss calculation
            nce_loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self.nce_weights, biases=self.nce_bias,
                labels=labels, inputs=input_embeddings,
                num_sampled=self.nce_num_sampled, num_classes=self.vocab_size,
                name="nce_loss"
            ), name="nce_loss_mean")
            self.add_loss(nce_loss)
        return input_embeddings

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'embeddings_size': self.embeddings_size,
            'symmetric_embeddings': self.symmetric_embeddings,
            'nce_num_sampled': self.nce_num_sampled,
            'embeddings_initializer': self.embeddings_initializer,
            'embeddings_regularizer': self.embeddings_regularizer,
            'embeddings_constraint': self.embeddings_constraint,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint
        }
        base_config = super(EmbeddingNCELayer, self).get_config()
        return dict(list(base_config.items()) + list(config))
