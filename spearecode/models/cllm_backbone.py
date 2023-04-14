import tensorflow as tf

from spearecode.layers.transformer_decoder import TransformerDecoder
from spearecode.layers.transformer_encoder import TransformerEncoder


class CLLM(tf.keras.Model):
    """A Code Large Language Model (CLLM) that implements the basic AlphaCode architecture for code generation tasks.

    Args:
        encoder_kwargs (dict): The keyword arguments for the TransformerEncoder.
        decoder_kwargs (dict): The keyword arguments for the TransformerDecoder.

    Attributes:
        encoder (TransformerEncoder): The Transformer encoder for the CLLM.
        decoder (TransformerDecoder): The Transformer decoder for the CLLM.

    Raises:
        ValueError: If the input has rank other than 1 or 2.

    """
    def __init__(self, *, encoder_kwargs, decoder_kwargs, **kwargs):
        """Initializes the CLLM with the specified hyperparameters."""
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_kwargs, **kwargs)
        self.decoder = TransformerDecoder(**decoder_kwargs, **kwargs)

    def __build(self):
        """Builds the model."""
        # Create a dummy input to build the model
        dummy_input = (tf.zeros((None, self.encoder_kwargs["encoder_context_len"])),
                       tf.zeros((None, self.decoder_kwargs["decoder_context_len"])))
        self.call(dummy_input)

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
        encoder_input, decoder_input = inputs

        # Get the encoder outputs that will be used in the decoder as well as for MLM logits for the MLM loss
        encoded_context, mlm_logits = self.encoder(encoder_input, **kwargs)  # (batch_size, context_len, d_model)

        # Take the encoder outputs and the decoder inputs and pass them through the decoder
        # to get the AR logits for calculating the AR loss
        ar_logits = self.decoder(decoder_input, encoded_context, **kwargs)  # (batch_size, target_len, d_model)

        # Return the final outputs
        return ar_logits, mlm_logits