import tensorflow as tf

from spearecode.layers.transformer_decoder import TransformerDecoder
from spearecode.layers.transformer_encoder import TransformerEncoder


class CLLM(tf.keras.Model):
    """A Code Large Language Model (CLLM) that implements the basic AlphaCode architecture for code generation tasks.

    Args:
        vocab_size (int): The size of the vocabulary used for tokenization.
        context_length (int): The length of the input context for the model.
        embedding_size (int): The size of the embedding vectors for each token.
        n_heads (int): The number of attention heads to use in the model.
        n_layers (int): The number of encoder and decoder layers to use in the model.
        use_bias (bool, optional): Whether or not to include bias terms in the layers.
        ffn_act (str, optional): The activation function to use in the feedforward layers. Can be "gelu" or "relu".
        expansion_factor (int, optional): The expansion factor to use in the feedforward layers.
        dropout_rate (float, optional): The rate of dropout to use in the model.

    Attributes:
        encoder (TransformerEncoder): The Transformer encoder for the CLLM.
        decoder (TransformerDecoder): The Transformer decoder for the CLLM.

    Raises:
        ValueError: If the input has rank other than 1 or 2.

    """
    def __init__(self, *, vocab_size, context_length, embedding_size, n_heads, n_layers,
                 use_bias=False, ffn_act="gelu", expansion_factor=4, dropout_rate=0.1):
        """Initializes the CLLM with the specified hyperparameters."""
        super(CLLM).__init__()
        transformer_kwargs = dict(
            vocab_size=vocab_size, context_length=context_length, embedding_size=embedding_size,
            n_heads=n_heads, n_layers=n_layers, use_bias=use_bias, ffn_act=ffn_act,
            expansion_factor=expansion_factor, dropout_rate=dropout_rate
        )
        self.encoder = TransformerEncoder(**transformer_kwargs)
        self.decoder = TransformerDecoder(**transformer_kwargs)

    def call(self, inputs, **kwargs):
        """Runs a forward pass of the CLLM.

        Args:
            inputs: The input to the model. Must be a tensor of shape (batch_size, context_length + target_length)
                or (batch_size, target_length).

        Returns:
            A tuple of the autoregressive logits and the MLM logits.

        Raises:
            ValueError: If the input has rank other than 1 or 2.
        """
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        encoder_input, decoder_input = self.unpack_tf_call_inputs(inputs)

        # Get the encoder outputs that will be used in the decoder as well as for MLM logits for the MLM loss
        encoded_context, mlm_logits = self.encoder(encoder_input, **kwargs)  # (batch_size, context_len, d_model)

        # Take the encoder outputs and the decoder inputs and pass them through the decoder
        # to get the AR logits for calculating the AR loss
        ar_logits = self.decoder(decoder_input, encoded_context, **kwargs)  # (batch_size, target_len, d_model)

        # Return the final outputs
        return ar_logits, mlm_logits

    @staticmethod
    def unpack_tf_call_inputs(inputs):
        """Unpacks the input tensor for use in the model.

        Args:
            inputs: The input tensor to unpack.

        Returns:
            A tuple of the encoder input tensor and the decoder input tensor.

        Raises:
            ValueError: If the input has rank other than 1 or 2.
        """
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        if len(tf.shape(inputs)) == 1:
            encoder_input, decoder_input = None, inputs
        elif len(tf.shape(inputs)) == 2:
            encoder_input, decoder_input = inputs
        else:
            raise ValueError(f"Expected inputs to have rank 1 or 2, but got rank {len(tf.shape(inputs))}.")
        return encoder_input, decoder_input
