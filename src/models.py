"""Model builders for the sequence-based DL models: plain LSTM, plain CNN, and
the CNN-LSTM(+attention) hybrid, plus a Keras Tuner hypermodel wrapper.

All three share the multi-task head (two outputs: dbt, wbt) and the same
regularization stack (L2 + Dropout), matching the brief's requirement that
EarlyStopping/ModelCheckpoint/Dropout/L2 are core anti-overfitting measures on
every DL model, not just the final one. Kept deliberately small/shallow — see
`reports/phase2_design.md` §4 for why, and each notebook prints `model.summary()`
so the parameter count vs. the 210-row training set is checked explicitly rather
than assumed.
"""
from __future__ import annotations

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def _output_heads(x: tf.Tensor, l2: float) -> list[tf.Tensor]:
    dbt = layers.Dense(1, name="dbt", kernel_regularizer=regularizers.l2(l2))(x)
    wbt = layers.Dense(1, name="wbt", kernel_regularizer=regularizers.l2(l2))(x)
    return [dbt, wbt]


def build_lstm(input_shape, units=16, dropout=0.3, l2=1e-3, dense_units=8) -> tf.keras.Model:
    """Plain LSTM baseline — no convolution, no attention. Ablation reference
    for what the CNN and attention pieces of the hybrid actually add."""
    inputs = layers.Input(shape=input_shape, name="sequence")
    x = layers.LSTM(units, kernel_regularizer=regularizers.l2(l2),
                     recurrent_regularizer=regularizers.l2(l2))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = _output_heads(x, l2)
    return tf.keras.Model(inputs, outputs, name="lstm_baseline")


def build_cnn(input_shape, filters=16, kernel_size=2, dropout=0.3, l2=1e-3, dense_units=8) -> tf.keras.Model:
    """Plain CNN baseline — convolution over the 4-step "temporal image" with no
    recurrence. Ablation reference for what the LSTM piece of the hybrid adds."""
    inputs = layers.Input(shape=input_shape, name="sequence")
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = _output_heads(x, l2)
    return tf.keras.Model(inputs, outputs, name="cnn_baseline")


def build_cnn_lstm(
    input_shape,
    filters=16,
    kernel_size=2,
    lstm_units=16,
    attention_key_dim=8,
    dropout=0.3,
    l2=1e-3,
    dense_units=8,
    use_attention=True,
) -> tf.keras.Model:
    """The primary model: Conv1D extracts local cross-channel patterns per short
    time window ("temporal image"), an optional single-head self-attention block
    (added per the Phase 1 JD finding that attention/Transformer-style
    architectures are far more evidenced than CNN-LSTM by name — see
    reports/phase1_jd_analysis.md) lets the model weigh which of the 4 lag steps
    matters most, and an LSTM models sequential dependence across what's left.
    Both output heads (dbt, wbt) share this backbone — the multi-task framing
    that distinguishes this from the internship's two independent models.
    """
    inputs = layers.Input(shape=input_shape, name="sequence")
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.Dropout(dropout)(x)

    if use_attention:
        attn_out = layers.MultiHeadAttention(
            num_heads=1, key_dim=attention_key_dim, kernel_regularizer=regularizers.l2(l2),
        )(query=x, value=x, key=x)
        x = layers.LayerNormalization()(x + attn_out)  # residual + norm, standard Transformer-lite block

    x = layers.LSTM(lstm_units, kernel_regularizer=regularizers.l2(l2),
                     recurrent_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = _output_heads(x, l2)
    return tf.keras.Model(inputs, outputs, name="cnn_lstm_hybrid")


OPTIMIZER_BUILDERS = {
    "adam": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
    "sgd": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    "rmsprop": lambda lr: tf.keras.optimizers.RMSprop(learning_rate=lr),
}


def compile_model(model: tf.keras.Model, optimizer: str = "adam", lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=OPTIMIZER_BUILDERS[optimizer](lr),
        loss={"dbt": "mse", "wbt": "mse"},
        metrics={"dbt": ["mae"], "wbt": ["mae"]},
    )
    return model


class CNNLSTMHyperModel(kt.HyperModel):
    """Keras Tuner search space for the CNN-LSTM hybrid.

    Deliberately narrow given ~210 training rows (the brief's own instruction:
    "constrain the search space appropriately for a small dataset, avoid
    tuning-induced overfitting"). Every choice here is a value the ablation
    notebook (04) already found plausible by hand, not a wide blind search —
    the search space would itself overfit the validation split if made much
    wider than this given only 46 validation rows.
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        filters = hp.Choice("filters", [8, 16, 24])
        kernel_size = hp.Choice("kernel_size", [2, 3])
        lstm_units = hp.Choice("lstm_units", [8, 16, 24])
        dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
        l2 = hp.Choice("l2", [1e-4, 1e-3, 1e-2])
        lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-2])
        use_attention = hp.Boolean("use_attention", default=True)

        model = build_cnn_lstm(
            self.input_shape, filters=filters, kernel_size=kernel_size,
            lstm_units=lstm_units, dropout=dropout, l2=l2, use_attention=use_attention,
        )
        return compile_model(model, optimizer="adam", lr=lr)
