import tensorflow as tf
from spearecode.layers.transformer_components import PositionalEmbedding, DecoderLayer
from spearecode.misc.misc_utils import prevent_mask_scaling


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer-based decoder layer for autoregressive sequence generation.

    This layer consists of a stack of `n_layers` decoder layers, each of which
    applies a causal self-attention mechanism followed by a cross-attention
    mechanism, and concludes with a feedforward network. The final output of
    the stack of layers is a dense layer producing the autoregressive logits.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        context_length (int): The length of the input sequences to the encoder.
        embedding_size (int): The dimensionality of the token embeddings.
        n_heads (int): The number of attention heads to use in each attention mechanism.
        n_layers (int): The number of decoder layers to stack.
        use_bias (bool): Whether or not to include bias terms in the layer computations. Defaults to False.
        ffn_act (str): The activation function to use in the feedforward network. Defaults to "gelu".
        expansion_factor (int): The expansion factor to use in the feedforward network. Defaults to 4.
        dropout_rate (float): The rate at which to apply dropout regularization. Defaults to 0.1.

    Attributes:
        embedding_size (int): The dimensionality of the token embeddings.
        n_layers (int): The number of decoder layers to stack.
        last_attn_scores: The attention scores from the last cross-attention mechanism
            computed by the layer (if any), which can be useful for visualization.
    """
    def __init__(self, *, vocab_size, context_length, embedding_size, n_heads, n_layers,
                 use_bias=False, ffn_act="gelu", expansion_factor=4, dropout_rate=0.1):
        super().__init__()

        self.supports_masking = True

        # Store arguments as attributes for later use
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.last_attn_scores = None

        # Store the decoder layer arguments for later use
        self.dec_kwargs = dict(
            embedding_size=embedding_size, n_heads=n_heads, use_bias=use_bias, ffn_act=ffn_act,
            expansion_factor=expansion_factor, dropout_rate=dropout_rate,
        )

        # Initialize positional embedding for the input tokens
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, embedding_size=embedding_size, context_length=context_length
        )

        # Initialize the decoder layers
        self.dec_layers = [DecoderLayer(**self.dec_kwargs) for _ in range(n_layers)]

        # Initialize dropout for the input
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Initialize the final dense layer for autoregressive output
        self.ar_head = tf.keras.layers.Dense(
            vocab_size, use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )

    def call(self, x, encoded_context=None, **kwargs):
        """ Applies the layer to the input `x`.

        If `encoded_context` is provided, the layer also uses a cross-attention
        mechanism to incorporate the encoded context into the decoding process.
        Returns the autoregressive logits produced by the layer.

        Args:
            x (tf.Tensor):
                – The input tensor of token IDs with shape (batch, target_seq_len).
            encoded_context (tf.Tensor, optional):
                – The output of the encoder with shape (batch, context_seq_len, embedding_size).

            **kwargs:
                – Additional keyword arguments to pass to the layer.

        Returns:
            tf.Tensor: The autoregressive logits produced by the layer with shape (batch, target_seq_len, vocab_size).
        """
        # `x` is token-IDs shape (batch, target_seq_len)
        # `encoded_context` is the output of the encoder (batch, context_seq_len, embedding_size)

        # Apply positional embedding to the input tokens
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        # Apply dropout to the input
        x = self.dropout(x, **kwargs)

        # Pass the input through the decoder layers
        for dec_layer in self.dec_layers:
            x = dec_layer(x, encoded_context, **kwargs)

        # Store the last attention scores from the final decoder layer
        #    --> Note that this will be None if no encoded_context is provided
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # Apply the final dense layer to produce autoregressive logits
        ar_logits = self.ar_head(x)  # (batch_size, target_seq_len, vocab_size)

        # Prevent scaling of logits due to masked positions
        # ar_logits = prevent_mask_scaling(ar_logits)

        return ar_logits
