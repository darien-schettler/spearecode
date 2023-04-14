import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    """Returns the positional encoding for a given length and depth.

    Args:
        length (int):
            - The length of the sequence.
        depth (int):
            - The number of dimensions in the positional encoding.

    Returns:
        A tensor of shape `(length, depth)` representing the positional encoding.
    """
    depth = depth // 2  # Ensure depth is an integer

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """ A tf.keras layer for computing and adding positional embeddings to token embeddings.

    Args:
        vocab_size (int):
            - The size of the vocabulary.
        embedding_size (int):
            - The size of the token embeddings.
        context_length (int):
            - The maximum length of the input sequence.

    Attributes:
        embedding (tf.keras.layers.Embedding):
            - The token embedding layer.
        pos_encoding (tf.Tensor):
            - The positional encoding tensor.
    """

    def __init__(self, vocab_size, embedding_size, context_length, **kwargs):
        super().__init__()
        self.embedding_size = embedding_size

        # Initialize the token embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)

        # Generate the positional encoding
        self.pos_encoding = positional_encoding(length=context_length, depth=embedding_size)

    def compute_mask(self, *args, **kwargs):
        """ Computes the mask for the input tensor.

        Returns:
            The mask for the input tensor.
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, **kwargs):
        """
        Passes the input tensor through the positional embedding layer.

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length)` representing the input sequence.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size + pos_encoding_size)` representing the
            input sequence with positional embeddings added.
        """
        length = tf.shape(x)[1]

        # Apply the token embedding layer
        x = self.embedding(x)

        # Scale the token embeddings and add the positional encoding
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    """ A tf.keras layer to use as a base for computing flavours of multi-head attention.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention):
            - The multi-head attention layer.
        add (tf.keras.layers.Add):
            - The add layer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()

    def call(self, x, *args, **kwargs):
        """
        Passes the input tensor through the multi-head attention layer.

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length, embedding_size)` representing the input sequence.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size)` representing the output of the
            multi-head attention layer.
        """
        raise NotImplementedError


class CrossAttention(BaseAttention):
    """ A tf.keras layer for computing cross-attention.

    Args:
        kwargs:
            - Additional arguments to pass to the multi-head attention layer.

    Attributes:
        last_attn_scores (tf.Tensor):
            - The attention scores from the last forward pass.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attn_scores = None

    def call(self, x, context=None, **kwargs):
        """ Passes the input tensor through the cross-attention layer.

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length, embedding_size)` representing the input sequence.
            context (tf.Tensor, optional):
                - A tensor of shape `(batch_size, context_length, embedding_size)` representing the context.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size)` representing the output of the
            cross-attention layer.
        """
        # Perform the cross attention and update the input with the attention output
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True, **kwargs)
        x = self.add([x, attn_output])

        # Cache the attention scores for plotting later
        self.last_attn_scores = attn_scores

        return x


class EncoderSelfAttention(BaseAttention):
    """ A tf.keras layer for computing self-attention in an encoder.

    Args:
        kwargs:
            - Additional arguments to pass to the multi-head attention layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        """Passes the input tensor through the self-attention

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length, embedding_size)` representing the input sequence.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size)` representing the output of the
            self-attention layer.
        """
        # Perform the self attention and update the input with the attention output
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        return x


class DecoderSelfAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        """ Passes the input tensor through the self-attention layer in a decoder.

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length, embedding_size)` representing the input sequence.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size)` representing the output of the
            self-attention layer.
        """

        # Perform the causal self attention and update the input with the attention output
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        return x


class FeedForwardNetwork(tf.keras.layers.Layer):
    """
    A feedforward neural network composed of:
        - A fully connected compression (squeeze)   layer followed by a GELU activation
        - A fully connected projection  (expansion) layer followed
        - A dropout layer.

    Args:
        embedding_size (int):
            – The number of embeddings.
        use_bias (bool, optional):
            – Whether to use bias in the dense layers.
        dropout_rate (float, optional):
            – The dropout rate to apply to the output of the dense residual layers.
        ffn_act (str, optional):
            – The activation function to use in the feedforward network.
        fc_expansion_factor (int, optional):
            – The expansion factor of the first dense layer.

    Attributes:
        ffn_act (tf.keras.layers.Layer):
            – The activation function corresponding to the ffn_act argument (default gelu).
        expansion_factor (int):
            – The expansion factor of the first dense layer.
        fc_expansion_dim (int):
            – The dimensionality of the expanded features.
        fc_expansion (tf.keras.layers.Layer):
            – A dense layer with GELU activation that expands the input features.
        fc_projection (tf.keras.layers.Layer):
            – A dense layer that reduces the dimensionality of the expanded features back to the original size.
        fc_dropout (tf.keras.layers.Layer):
            – A dropout layer that applies dropout to the output of the dense layers.
    """

    def __init__(self, embedding_size, use_bias=False, dropout_rate=0.1, ffn_act="gelu", fc_expansion_factor=4):
        super(FeedForwardNetwork, self).__init__()

        # Calculated attribute values for our layers
        self.ffn_act = tf.keras.activations.get(ffn_act)
        self.expansion_factor = fc_expansion_factor
        self.fc_expansion_dim = self.expansion_factor * embedding_size

        # The fully connected layer that expands the input features (default factor of 4)
        self.fc_expansion = tf.keras.layers.Dense(
            self.fc_expansion_dim,
            activation=self.ffn_act,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )

        # The fully connected layer that reduces the dimensionality of the expanded features back to the original size
        self.fc_projection = tf.keras.layers.Dense(
            embedding_size,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )
        self.fc_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None, **kwargs):
        """ Passes the input tensor through the feedforward network.

        Args:
            x (tf.Tensor):
                – A tensor of shape `(batch_size, seq_length, hidden_size)`.
            training (bool, optional):
                – Whether the model is in training mode.

        Returns:
            A tensor of the same shape as the input tensor.
        """

        # Pass the input through the layers
        x = self.fc_expansion(x)
        x = self.fc_projection(x)
        x = self.fc_dropout(x, training=training)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """ The monomer encoder layer that makes up the EncoderTransformer

    Args:
        embedding_size (int):
            - The size of the embeddings.
        n_heads (int):
            - The number of attention heads.
        use_bias (bool, optional):
            - Whether to use bias in the dense layers.
        ffn_act (str, optional):
            - The activation function to use in the feedforward network.
        expansion_factor (int, optional):
            - The expansion factor of the first dense layer.
        dropout_rate (float, optional):
            - The dropout rate to apply to the output of the dense residual layers.
    """
    def __init__(self, *, embedding_size, n_heads,
                 use_bias=False, ffn_act="gelu",
                 expansion_factor=4, dropout_rate=0.1):
        super().__init__()

        self.self_attention = EncoderSelfAttention(
            num_heads=n_heads, key_dim=embedding_size, dropout=dropout_rate, use_bias=use_bias
        )
        self.ffn = FeedForwardNetwork(
            embedding_size=embedding_size, dropout_rate=dropout_rate,
            fc_expansion_factor=expansion_factor,
            ffn_act=ffn_act, use_bias=use_bias
        )
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, **kwargs):
        x = self.self_attention(self.attn_layer_norm(x), **kwargs)
        x = self.ffn(self.ffn_layer_norm(x), **kwargs)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    """ Passes the input tensor through the encoder layer.

    Args:
        embedding_size (int):
            - The size of the embeddings.
        n_heads (int):
            - The number of attention heads.
        use_bias (bool, optional):
            - Whether to use bias in the dense layers.
        ffn_act (str, optional):
            - The activation function to use in the feedforward network.
        expansion_factor (int, optional):
            - The expansion factor of the first dense layer.
        dropout_rate (float, optional):
            - The dropout rate to apply to the output of the dense residual layers.
    """
    def __init__(self, *, embedding_size, n_heads,
                 use_bias=False, ffn_act="gelu",
                 expansion_factor=4, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.last_attn_scores = None
        self.causal_self_attention = DecoderSelfAttention(
            num_heads=n_heads, key_dim=embedding_size, dropout=dropout_rate, use_bias=use_bias
        )

        self.cross_attention = CrossAttention(
            num_heads=n_heads, key_dim=embedding_size, dropout=dropout_rate, use_bias=use_bias
        )

        self.ffn = FeedForwardNetwork(
            embedding_size=embedding_size, dropout_rate=dropout_rate,
            fc_expansion_factor=expansion_factor,
            ffn_act=ffn_act, use_bias=use_bias
        )

    def call(self, x, context=None, **kwargs):
        """ Passes the input tensor through the decoder layer.

        Args:
            x (tf.Tensor):
                - A tensor of shape `(batch_size, seq_length, embedding_size)` representing the input sequence.
            context (tf.Tensor, optional):
                - A tensor of shape `(batch_size, context_length, embedding_size)` representing the context sequence.

        Returns:
            A tensor of shape `(batch_size, seq_length, embedding_size)` representing the output of the
            decoder layer.
        """
        x = self.causal_self_attention(x=x)
        if context is not None:
            # Include the cross attention layer if the context is provided
            # and update the last attention scores for plotting later
            x = self.cross_attention(x=x, context=context, **kwargs)
            self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x, **kwargs)  # Shape `(batch_size, seq_len, d_model)`.
        return x
