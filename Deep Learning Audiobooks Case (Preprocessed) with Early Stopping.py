import numpy as np
import tensorflow as tf
from sklearn import preprocessing
# I used the sklearn preprocessing library to standardize the data more easily.

raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1]
targets_all = raw_csv_data[:, -1]
# Except for the first column (customer IDs that bear no useful information) and the last column (targets), the inputs are all columns in the csv.

shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)
# Because it was actually arranged by date, I shuffled the indices of the data so that it is not biased when I feed it into the model.
# I also batched the data, so I want it to be as randomly spread out as possible

unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
targets_all = targets_all[shuffled_indices]
# Then I shuffled the inputs and targets using the shuffled indices. 

num_one_targets = int(np.sum(targets_all))
# To count how many targets are 1 (customers that converted)
zero_targets_counter = 0
# Here I set a counter for targets that are 0 (customers that did not convert)

indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
# I removed some input/target pairs to create a balanced dataset
# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, I marked entries where the target is 0.
# I will remove these marked entries/indices below to create a 50-50 dataset.
            
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
# I created two new variables, one that will contain the inputs, and one that will contain the targets.

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
# I took advantage of the preprocessing capability of sklearn here.

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
# I shuffled the preprocessed data, inputs, and targets to prepare them for training, validation, and testing.

samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
# Here I counted the samples in each subset and aimed for an 80-10-10 distribution of training, validation, and testing.

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
# These are the variables I created that recorded the inputs and targets for training.

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
# These are the variables I created that recorded the inputs and targets for validation.

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
# These are the variables I created that recorded the inputs and targets for testing.

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
# Here I wanted to make sure that the training, validation, and testing data are balanced like my dataset (targets 0 and 1 are 50-50).
# I printed the number of targets that are 1s, the total number of samples, and the proportion of training, validation, and test.

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
# I saved the three datasets in 3 .npz files, with coherent filenames.

npz = np.load('Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)
# I wanted to make sure that all my inputs are floats.
# I wanted my targets to be integers so I can smoothly one-hot encode them with sparse_categorical_crossentropy.

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
# This is the validation data in a temporary variable

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
# This is the test data in a temporary variable

# Training the model
input_size = 10
output_size = 2
hidden_layer_size = 50
    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
    # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 
    # 2nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax')
    # output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
# Here I set an early stopping mechanism with a patience=3 for tolerance against random validation loss increases.

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size, 
          epochs=max_epochs, 
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2
          # To get enough information about the training process
          )  

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
