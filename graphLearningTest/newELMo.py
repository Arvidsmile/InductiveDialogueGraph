import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
# Run this code once -------------------------------
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable = False)
  embedded_text = elmo(text_input, signature="default", as_dict=True)["elmo"]
  output = tf.reduce_mean(embedded_text,1)
  init_op = tf.group([tf.compat.v1.global_variables_initializer(),
                      tf.compat.v1.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.compat.v1.Session(graph=g)
session.run(init_op)
#-------------------------------------------------------

# ---- inference time!
result = session.run(output, feed_dict={text_input: ["My first sentence",
                                                     "My second sentence",
                                                     "Ice cream is tasty!"]})

# ---- or as a function (that can be put into a loop using the batches trick)
def elmo_vectors2(session, tf_output, tf_input, x):
  result = session.run(tf_output, feed_dict={tf_input: x.tolist()})
  return result

print(result.shape)
print(result)