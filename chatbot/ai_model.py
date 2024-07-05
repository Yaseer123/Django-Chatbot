# gpt2_model.py
import os
import time
import keras_nlp
import tensorflow as tf

# Set environment variable for Keras backend
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

# Set global policy for mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

# Load preprocessor and GPT-2 model
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

def generate_response(message):
    # Generate text based on the input message
    output = gpt2_lm.generate(message, max_length=200)
    generated_response = "\n".join(output)
    return generated_response
