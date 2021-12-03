
from keras import initializers
from keras import backend as K
from keras.utils import tf_inspect
from keras.mixed_precision import policy
from tensorflow.keras.layers import InputSpec, Layer

class AttentionWeightedAverage(Layer):
    """
    Class extracted and adapted from module virnet/AttentionLayer.py of
    VirNet package [Abdelkareem et al. 2018]

    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.initializer = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__()

    def get_config(self):
        """
        Method adapted from module keras/keras/engine/base_layer.py of
        Keras API to work with custom layer cloning
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {
            'name': self.name,
            'trainable': self.trainable,
            'initializer': self.initializer,
            'supports_masking': self.supports_masking,
            'return_attention': self.return_attention
        }
        if hasattr(self, '_batch_input_shape'):
          config['batch_input_shape'] = self._batch_input_shape
        config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, 'dynamic'):
          # Only include `dynamic` in the `config` if it is `True`
          if self.dynamic:
            config['dynamic'] = self.dynamic
          elif 'dynamic' in all_args:
            all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
          raise NotImplementedError(textwrap.dedent(f"""
              Layer {self.__class__.__name__} has arguments {extra_args}
              in `__init__` and therefore must override `get_config()`.
              Example:
              class CustomLayer(keras.layers.Layer):
                  def __init__(self, arg1, arg2):
                      super().__init__()
                      self.arg1 = arg1
                      self.arg2 = arg2
                  def get_config(self):
                      config = super().get_config()
                      config.update({{
                          "arg1": self.arg1,
                          "arg2": self.arg2,
                      }})
                      return config"""))

        return config

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.initializer)
        self.train_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
