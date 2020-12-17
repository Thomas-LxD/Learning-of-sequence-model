import tensorflow as tf
from tensorflow.keras import layers


if __name__ =='__main__':
    """功能型API使用共享层"""

    # Embedding for 1000 unique words mapped to 128-dimensional vectors
    share_embedding = layers.Embedding(1000, 128)

    text_input_a = tf.keras.Input(shape=(None, ), dtype='int32')
    text_input_b = tf.keras.Input(shape=(None, ), dtype='int32')

    encoded_input_a = share_embedding(text_input_a)
    encoded_input_b = share_embedding(text_input_b)


