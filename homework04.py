import tensorflow_datasets as tfds
import tensorflow as tf

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# First step:

def preprocess(mnist):
  #flatten the images into vectors
  mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
  #convert data from uint8 to float32
  mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
  mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
  mnist = mnist.cache()
  #shuffle, batch, prefetch
  mnist = mnist.shuffle(1000)
  mnist = mnist.batch(32)
  mnist = mnist.prefetch(20)
  #return preprocessed dataset
  return mnist

train_dataset = train_ds.apply(preprocess)
test_dataset = test_ds.apply(preprocess)

# Second step:

zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), data.shuffle(2000)))
# map ((x1,y1),(x2,y2)) to (x1,x2, y1==y2*) *boolean
zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], x1[1]==x2[1]))
# transform boolean target to int
zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.int32)))
# batch the dataset
zipped_ds = zipped_ds.batch(batch_size)
# prefetch
zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
return zipped_ds

#zusammenfügen mit der ersten funktion?