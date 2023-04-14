import tensorflow as tf
from spearecode.layers.transformer_components import PositionalEmbedding, EncoderLayer
from spearecode.misc.misc_utils import prevent_mask_scaling


class TransformerEncoder(tf.keras.layers.Layer):
    """A Transformer-based encoder layer for masked language modeling.

        Args:
            vocab_size (int): The size of the vocabulary.
            context_length (int): The length of the input sequence.
            embedding_size (int): The size of the embedding dimension.
            n_heads (int): The number of heads for the multi-head attention layer.
            n_layers (int): The number of encoder layers.
            use_bias (bool, optional): Whether to use bias in the dense layers. Default is False.
            ffn_act (str, optional): The activation function for the feedforward network. Default is "gelu".
            expansion_factor (int, optional): The expansion factor for the feedforward network. Default is 4.
            dropout_rate (float, optional): The dropout rate to use. Default is 0.1.

        Attributes:
            enc_kwargs (dict): The arguments for initializing each encoder layer.
            pos_embedding (PositionalEmbedding): The positional embedding layer for the input tokens.
            enc_layers (list): The list of initialized encoder layers.
            dropout (Dropout): The dropout layer for the input.
            mlm_head (Dense): The final dense layer for masked language model output.
        """
    def __init__(self, vocab_size, context_length, embedding_size, n_heads, n_layers,
                 use_bias=False, ffn_act="gelu", expansion_factor=4, dropout_rate=0.1):
        super().__init__()

        self.supports_masking = True

        # Store the encoder layer arguments for later use
        self.enc_kwargs = dict(
            embedding_size=embedding_size, n_heads=n_heads, use_bias=use_bias, ffn_act=ffn_act,
            expansion_factor=expansion_factor, dropout_rate=dropout_rate
        )

        # Initialize positional embedding for the input tokens
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, embedding_size=embedding_size, context_length=context_length)

        # Initialize the encoder layers
        self.enc_layers = [EncoderLayer(**self.enc_kwargs) for _ in range(n_layers)]

        # Initialize dropout for the input
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Initialize the final dense layer for masked language model output
        self.mlm_head = tf.keras.layers.Dense(
            vocab_size, use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )

    def call(self, x, **kwargs):
        """Forward pass of the TransformerEncoder layer.

         Args:
             x (tf.Tensor): The input tensor of shape `(batch_size, seq_len)`.

         Returns:
             tuple: A tuple of the encoded input tensor of shape `(batch_size, n_context, embedding_size)`
                 and the masked language model logits of shape `(batch_size, context_len, vocab_size)`.
         """
        # `x` is token-IDs shape: (batch, seq_len)

        # Apply positional embedding to the input tokens
        x = self.pos_embedding(x)  # Shape `(batch_size, n_context, embedding_size)`.

        # Apply dropout to the input
        x = self.dropout(x)

        # Pass the input through the encoder layers
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        # Additional layer to capture mlm logits for calculating Encoder only loss
        mlm_logits = self.mlm_head(x)  # (batch_size, context_len, vocab_size)

        # Prevent scaling of logits due to masked positions
        mlm_logits = prevent_mask_scaling(mlm_logits)

        # Shape `(batch_size, n_context, embedding_size)`.
        return x, mlm_logits


