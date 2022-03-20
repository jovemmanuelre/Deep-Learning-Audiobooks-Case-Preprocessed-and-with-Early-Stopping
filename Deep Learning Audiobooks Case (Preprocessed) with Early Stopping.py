from sklearn import preprocessing
import numpy as np
import tensorflow as tf

raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1]
# I excluded the customer IDs from my inputs because they aren't useful in my model

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
# Here I balanced my dataset prevent the algorithm from giving an unbalanced output.

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
# I shuffled the indices of the data, so the data is not arranged in any way once I feed it into the algorithm.

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
# I used the shuffled indices to shuffle my inputs and targets.
# I also spread out my data randomly to prepare it for batching.

samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
# Here I decided to distribute my inputs into 80-10-10: training, validation, and test.

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
# Created variables that record the inputs and targets for training.
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
# Created variables that record the inputs and targets for validation.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
# Created variables that record the inputs and targets for test.

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
# Here I wanted to check whether the data for training, validation, and test that were taken from the shuffled dataset are balanced, like the data for my targets.

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)

npz = np.load('Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(float)
train_targets = npz['targets'].astype(int)
# I wanted to make sure that all my inputs are floats.
# I wanted my targets to be integers,so I can smoothly one-hot encode them with sparse_categorical_crossentropy.

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(float), npz['targets'].astype(int)
# This is the validation data in a temporary variable.

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(float), npz['targets'].astype(int)
# This is the test data in a temporary variable.


# Training the model

input_size = 10
output_size = 2
# Ouput size is 2 because I hot-encoded the categorical variables.
hidden_layer_size = 50
    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
# Here I set an early stopping mechanism with a patience=3 to be a bit tolerant against random validation loss increases.

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size, 
          epochs=max_epochs, 
          callbacks=[early_stopping],
          # callbacks are functions called when a task is completed to check if the validation loss is increasing.
          validation_data=(validation_inputs, validation_targets),
          verbose=2
          )  

print('End of training.')

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
