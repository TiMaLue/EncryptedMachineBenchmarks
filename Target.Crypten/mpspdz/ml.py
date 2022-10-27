from Compiler import ml
tf = ml

training_samples = sfix.Tensor([60, 28, 28])
training_labels = sint.Tensor([60, 10])

test_samples = sfix.Tensor([10, 28, 28])
test_labels = sint.Tensor([10, 10])

training_labels.input_from(0)
training_samples.input_from(0)

test_labels.input_from(0)
test_samples.input_from(0)

layers = [
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10,  activation='softmax')
]

model = tf.keras.models.Sequential(layers)

optim = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01)

model.compile(optimizer=optim)

opt = model.fit(
  training_samples,
  training_labels,
  epochs=1,
  batch_size=2,
  validation_data=(test_samples, test_labels)
)