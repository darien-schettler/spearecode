import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import re


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients.
    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.
    Instead we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 gradient_clip_norm=1.0,
                 name='AdamWeightDecay',
                 **kwargs):
        """ Initializes the AdamWeightDecay optimizer.

        Args:
            learning_rate (float, optional):
                - The learning rate.
            beta_1 (float, optional):
                - The exponential decay rate for the first moment estimates.
            beta_2 (float, optional):
                - The exponential decay rate for the second moment estimates.
            epsilon (float, optional):
                - A small constant for numerical stability.
            amsgrad (bool, optional):
                - Whether to apply AMSGrad variant of this algorithm from the paper "On the
                  Convergence of Adam and Beyond".
            weight_decay_rate (float, optional):
                - The weight decay rate.
            include_in_weight_decay (list, optional):
                - A list of regex patterns to include in weight decay computation.
            exclude_from_weight_decay (list, optional):
                - A list of regex patterns to exclude from weight decay computation.
            gradient_clip_norm (float, optional):
                - The maximum norm for gradient clipping.
            name (str, optional):
                - The name of the optimizer.
            kwargs:
                - Additional arguments passed to the base class.
        """
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self.gradient_clip_norm = gradient_clip_norm
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config, **kwargs):
        """ Creates an optimizer from its config with WarmUp custom object.

        Args:
            config:
                - The configuration to use for the optimizer.

        Returns:
            An instance of the optimizer.
        """
        custom_objects = {}
        return super(AdamWeightDecay, cls).from_config(
            config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """ Prepares the optimizer to apply weight decay.

        Args:
            var_device:
                - The device on which the variable is placed.
            var_dtype:
                - The data type of the variable.
            apply_state:
                - The state of the optimizer.

        Returns: None;
        """
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate'
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        """ Computes the operation to decay weights based on weight decay rate.

        Args:
            var:
                - The variable to apply weight decay on.
            learning_rate:
                - The learning rate to use for the update.
            apply_state:
                - The state of the optimizer.

        Returns:
            A TensorFlow operation to decay the weights.
        """
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
                use_locking=self._use_locking
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True, **kwargs):
        """ Applies gradients to variables.

        Args:
            grads_and_vars:
                - A list of gradient and variable pairs.
            name (str, optional):
                - The name of the operation.
            experimental_aggregate_gradients (bool, optional):
                - Whether to aggregate gradients. Default is True.

        Returns:
            An operation to apply gradients to variables.
        """
        grads, tvars = list(zip(*grads_and_vars))
        if experimental_aggregate_gradients and self.gradient_clip_norm > 0.0:
            # when experimental_aggregate_gradients = False, apply_gradients() no
            # longer implicitly allreduce gradients, users manually allreduce gradient
            # and passed the allreduced grads_and_vars. For now, the
            # clip_by_global_norm will be moved to before the explicit allreduce to
            # keep the math the same as TF 1 and pre TF 2.2 implementation.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients
        )

    def _get_lr(self, var_device, var_dtype, apply_state):
        """ Retrieves the learning rate with the given state.

        Args:
            var_device:
                - The device on which the variable is placed.
            var_dtype:
                - The data type of the variable.
            apply_state:
                - The state of the optimizer.

        Returns:
            The learning rate with the given state.
        """
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """ Applies dense gradients to the variable.

        Args:
            grad:
                - The gradients to apply.
            var:
                - The variable to apply gradients on.
            apply_state:
                - The state of the optimizer.

        Returns:
            An operation to apply the dense gradients to the variable.
        """
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """ Applies sparse gradients to the variable.

        Args:
            grad:
                - The gradients to apply.
            var:
                - The variable to apply gradients on.
            indices:
                - The indices to apply gradients on.
            apply_state (, optional):
                - The state of the optimizer.

        Returns:
            An operation to apply the sparse gradients to the variable.
        """
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        """Gets the configuration of the optimizer.

        Returns:
            A dictionary containing the configuration of the optimizer.
        """
        config = super(AdamWeightDecay, self).get_config()
        config.update({'weight_decay_rate': self.weight_decay_rate, })
        return config

    def _do_use_weight_decay(self, param_name):
        """ Checks whether to use L2 weight decay for a parameter.

        Args:
            param_name (str):
                - The name of the parameter.

        Returns:
            True if L2 weight decay is to be used, False otherwise.
        """
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate schedule with warm up.
    This learning rate schedule combines a linear warm up with a cosine decay.
    The learning rate is increased linearly from an initial learning rate to a
    maximum learning rate over a specified number of warm up steps. After the
    warm up period, the learning rate is decayed according to a cosine schedule
    over a specified number of decay steps.

    Args:
        init_lr (float):
            - The initial learning rate.
        decay_steps (int):
            - The number of steps over which to decay the learning rate.
        warmup_steps (int):
            - The number of steps over which to linearly increase the learning rate
              from the initial learning rate to the maximum learning rate.
        hold_steps (int, optional):
            - The number of steps to hold the learning rate constant at
              the maximum learning rate before starting the cosine decay.
        alpha (float, optional):
            - Minimum learning rate value as a fraction of init_lr
        min_lr (float, optional):
            - The minimum possible learning rate

    Returns:
        A tf.float32: The learning rate for the current step.
    """

    def __init__(self, init_lr, decay_steps, warmup_steps, hold_steps=0, alpha=0.0, min_lr=1e-9, **kwargs):
        super(WarmUpCosineDecay, self).__init__()

        # Set attributes
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.alpha = alpha

    def __call__(self, step):
        """Compute the learning rate for the given step.
        Args:
            step (int):
                – The current step.

        Returns:
            The learning rate for the current step.
        """
        global_step = tf.cast(step, dtype=tf.float32)

        # Compute warm up learning rate
        warmup_lr = self.init_lr * (global_step / self.warmup_steps)

        # Compute cosine decay learning rate
        #   --> 0.5 * lr * (1 + cos((pi*current_step) / (total_decay_steps)))
        #   --> assume warmup of 500, hold of 100, init_lr of 0.001, current step of 1000, total_decay of 5000
        #       -----> 0.5 * 0.001 * (1 + cos(pi*1000/5000)) = 0.00090450
        cosine_decay_lr = 0.5 * self.init_lr * (
                1.0 + tf.cos(
            tf.constant(math.pi, tf.float32) *
            tf.cast(global_step - self.warmup_steps - self.hold_steps, tf.float32) /
            tf.cast(self.decay_steps - self.warmup_steps - self.hold_steps, tf.float32)
        ))

        # If default this doesn't do anything
        cosine_decay_lr = (1 - self.alpha) * cosine_decay_lr + self.alpha

        # Choose between warm up and cosine decay
        if self.hold_steps > 0:
            learning_rate = tf.where(
                global_step < self.warmup_steps,
                tf.math.maximum(warmup_lr, self.min_lr),
                tf.where(
                    global_step < (self.warmup_steps + self.hold_steps),
                    self.init_lr,
                    tf.math.maximum(cosine_decay_lr, self.min_lr)
                )
            )
        else:
            learning_rate = tf.where(
                global_step < self.warmup_steps,
                tf.math.maximum(warmup_lr, self.min_lr),
                tf.math.maximum(cosine_decay_lr, self.min_lr)
            )

        return learning_rate

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            'init_lr': self.init_lr,
            'min_lr': self.min_lr,
            'decay_steps': self.decay_steps,
            'warmup_steps': self.warmup_steps,
            'hold_steps': self.hold_steps,
            'alpha': self.alpha,
        }


def plot_learning_rate(init_lr=1e-3, min_lr=1e-4, decay_steps=5_000, warmup_steps=500,
                       hold_steps=100, total_steps=5_000, _figsize=(18, 10), **kwargs):
    """Plots the learning rate schedule for a given set of parameters.

    Args:
        init_lr (float, optional):
            - The initial learning rate.
        min_lr (float, optional):
            - The initial learning rate.
        decay_steps (int, optional):
            - The number of steps over which to decay the learning rate.
        warmup_steps (int, optional):
            - The number of steps over which to linearly increase the lr
              from the initial lr to the maximum lr.
        hold_steps (int, optional):
            - The number of steps to hold the learning rate constant at
              the maximum learning rate before starting the cosine decay.
        total_steps (int, optional):
            - The number of steps to plot. Defaults to 1000.
        _figsize (tuple of ints, optional):
            - The size of the matplotlib.figure object

    Returns:
        None; Plot is generated in browser.

    Example/Default Usage:
        - This assumes a 5 epoch training over 100,000 total steps (20,000 steps/epoch).
        - The ramp up time is 1 epoch with a hold of a quarter of an epoch.
        - The following epochs will experience cosine decay at a rate of once per epoch
    """

    schedule = WarmUpCosineDecay(init_lr, decay_steps, warmup_steps, hold_steps, min_lr=min_lr)
    lrs = schedule(tf.constant(np.arange(total_steps)))

    plt.figure(figsize=_figsize)
    plt.plot(lrs)
    plt.title("Plot of Cosine Decay Learning Rate Schedule With Warmup/Hold", fontweight="bold")
    plt.xlabel('Step', fontweight="bold")
    plt.ylabel('Learning Rate', fontweight="bold")
    plt.show()