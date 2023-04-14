import tensorflow as tf
from spearecode.layers.transformer_encoder import TransformerEncoder
from spearecode.layers.transformer_decoder import TransformerDecoder


class DecoderOnlyBackbone(tf.keras.Model):
    """ A TensorFlow Keras model representing a decoder-only backbone for code generation tasks.

    This model consists of a `TransformerDecoder` layer that takes in a tokenized sequence and produces autoregressive
    logits for each token in the sequence. This model is the isolated component of the CLLM model which was
    designed to be used in conjunction with a MLM (masked language model) head for pre-training on code corpora

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        context_length (int): The maximum context length of the input sequence.
        embedding_size (int): The dimensionality of the token embeddings.
        n_heads (int): The number of attention heads to use in the multi-head attention layer.
        n_layers (int): The number of transformer layers in the model.
        use_bias (bool, optional): Whether to use bias vectors in the transformer layers. Defaults to False.
        ffn_act (str, optional): The activation function to use in the feedforward layer of the transformer
            layers. Defaults to "gelu".
        expansion_factor (int, optional): The expansion factor to use in the feedforward layer of the transformer
            layers. Defaults to 4.
        dropout_rate (float, optional): The rate of dropout to use in the transformer layers. Defaults to 0.1.

    Returns:
        A TensorFlow Keras model representing the decoder-only backbone of CLLM
    """
    def __init__(self, *, vocab_size, context_length, embedding_size, n_heads, n_layers,
                 use_bias=False, ffn_act="gelu", expansion_factor=4, dropout_rate=0.1):
        super(DecoderOnlyBackbone, self).__init__()
        self.transformer_decoder = TransformerDecoder(
            vocab_size=vocab_size, context_length=context_length, embedding_size=embedding_size,
            n_heads=n_heads, n_layers=n_layers, use_bias=use_bias, ffn_act=ffn_act,
            expansion_factor=expansion_factor, dropout_rate=dropout_rate
        )

    def call(self, x, training=None, **kwargs):
        ar_logits = self.transformer_decoder(x, training=training, **kwargs)
        return ar_logits


class EncoderOnlyBackbone(tf.keras.Model):
    """A TensorFlow Keras model representing an encoder-only backbone for code generation tasks.

    This model consists of a `TransformerEncoder` layer that takes in a tokenized sequence and produces an
    encoded output representation, which is used as input to a task-specific decoder. This model is the isolated
    component of the CLLM model which was designed to be used in conjunction with an Auto-Regressive head (decoder)
    for pre-training on code corpora.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        context_length (int): The maximum context length of the input sequence.
        embedding_size (int): The dimensionality of the token embeddings.
        n_heads (int): The number of attention heads to use in the multi-head attention layer.
        n_layers (int): The number of transformer layers in the model
        use_bias (bool, optional): Whether to use bias vectors in the transformer layers. Defaults to False.
        ffn_act (str, optional): The activation function to use in the feedforward layer of the transformer
            layers. Defaults to "gelu".
        expansion_factor (int, optional): The expansion factor to use in the feedforward layer of the transformer
            layers. Defaults to 4.
        dropout_rate (float, optional): The rate of dropout to use in the transformer layers. Defaults to 0.1.

    Returns:
        A TensorFlow Keras model representing the encoder-only backbone part of CLLM
    """

    def __init__(self, vocab_size, context_length, embedding_size, n_heads, n_layers,
                 use_bias=False, ffn_act="gelu", expansion_factor=4, dropout_rate=0.1):
        super(EncoderOnlyBackbone, self).__init__()

        self.transformer_encoder = TransformerEncoder(
            vocab_size=vocab_size, context_length=context_length, embedding_size=embedding_size,
            n_heads=n_heads, n_layers=n_layers, use_bias=use_bias, ffn_act=ffn_act,
            expansion_factor=expansion_factor, dropout_rate=dropout_rate
        )

    def call(self, x, training=None, **kwargs):
        encoded_output, mlm_logits = self.transformer_encoder(x, training=training, **kwargs)
        return encoded_output, mlm_logits
