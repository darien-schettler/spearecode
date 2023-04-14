import tensorflow as tf
import numpy as np


def get_callbacks(config):
    """ Returns a list of callbacks to be used during training.

    Args:
        config (CallbackConfig): The config object containing the callback parameters.

    Returns:
        callbacks (list): A list of callbacks to be used during training.
    """

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config["ckpt_dir"], verbose=config["verbose"], save_best_only=True, save_weights_only=config["save_weights_only"]
        )
    ]
    if config["use_early_stopping"]:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=config["es_patience"], verbose=config["verbose"], restore_best_weights=True
            )
        )
    return callbacks


class MaskedTextGenerator(tf.keras.callbacks.Callback):
    def __init__(self, sample_tokens, decoder, top_k=5, mask_token_id=4, pad_to=128, pad_constant=6):
        super().__init__()
        self.decoder = decoder
        self.pad_to = pad_to
        self.pad_constant = pad_constant
        self.sample_tokens = self.pad_to_len(sample_tokens)
        self.original_len = tf.shape(sample_tokens)[-1]
        self.mask_token_id = mask_token_id
        self.k = top_k

    def pad_to_len(self, arr):
        current_len = tf.shape(arr)[-1]
        pad_amount = self.pad_to - current_len
        pad_arr = tf.pad(arr, [(0, 0), (0, pad_amount)], mode='CONSTANT', constant_values=self.pad_constant)
        return pad_arr

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model(self.sample_tokens, training=False).numpy()[:, :self.original_len]
        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]
        top_indices = mask_prediction[0].argsort()[-self.k:][::-1]
        values = mask_prediction[0][top_indices]

        _input_text = self.decoder(self.sample_tokens[0].numpy()[:self.original_len])
        print("\n\n--- TEST MODEL PERFORMANCE ---")
        print(f"\tINPUT TEXT --> '{_input_text}'")

        print(f"\n--- TOP {len(top_indices)} PROBABLE INFERENCES ---")
        for i in range(len(top_indices)):
            v, p = values[i], top_indices[i]
            _tokens = np.copy(self.sample_tokens[0, :self.original_len])
            _tokens[masked_index[0]] = p
            print(f"\tOUR {i + 1}th PREDICTION --> '{self.decoder(_tokens)}'  ([mask]={self.decoder(p)}@{v:.4f})")
        print("\n")