import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class UseSimilarity(object):
    def __init__(self, use_dir):
        
        self.embed = hub.Module(use_dir)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def get_sim(self, sentences):
        message_embeddings = self.session.run(self.embed(sentences))
        arr = np.array(message_embeddings).tolist()
        print(np.inner(arr[0], arr[1]))

    def __exit__(self):
        self.session.close()

    def close(self):
        self.session.close()
