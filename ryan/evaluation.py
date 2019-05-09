import numpy as np
from keras.models import load_model
import keras.backend as K


def evaluate(model_path, test_set_path):
    # Load test set
    test_chunks = np.load(test_set_path + 'test_chunks.npy', allow_pickle=True)
    test_labels = np.load(test_set_path + 'test_labels.npy')

    # Load model.
    def exact_pred(y_true, y_pred):
        return K.min(K.cast(K.equal(y_true, K.round(y_pred)), dtype='float16'), axis=-1)

    def full_multi_label_metric(y_true, y_pred):
        comp = K.equal(y_true, K.round(y_pred))
        return K.cast(K.all(comp, axis=-1), K.floatx())

    model = load_model(model_path, custom_objects={"full_multi_label_metric": full_multi_label_metric})

    # For each set of chunks in test_chunks, make predictions for each of them.
    num_chunks = 0
    chunk_correct = 0
    predictions = []
    print(str(len(test_chunks)) + " chunk sets to classify...")
    for sample_index, chunk_set in enumerate(test_chunks):
        print("\r", "Classifying chunk set " + str(sample_index + 1), end="")

        # Tally the number of chunks we've seen.
        num_chunks += len(chunk_set)

        # Make a prediction for each of the chunks in chunk_set.
        # curr_predictions = model.predict(x=chunk_set)
        curr_predictions = np.empty((0, test_labels.shape[1]))
        for chunk in chunk_set:
            chunk_prediction = model.predict(x=chunk.reshape((1,) + chunk.shape + (1,)))[0]
            curr_predictions = np.concatenate((curr_predictions, chunk_prediction.reshape((1,) + chunk_prediction.shape)))

        # Tally the number of chunks which match the correct label.
        for p in curr_predictions:
            normalized_p = np.round(p)
            if list(normalized_p) == list(test_labels[sample_index]):
                chunk_correct += 1

        # Generate the resulting label vector for all chunks in this chunk set.
        res_label_vector = [0] * test_labels.shape[1]
        for i in range(len(res_label_vector)):
            if max(curr_predictions[:, i]) > 0.5:
                res_label_vector[i] = 1
        predictions.append(res_label_vector)

    # Compare resulting label vectors against target label vectors in test_labels.
    correct = 0
    num_multi_label_correct = 0
    num_multi_label_samples = 0
    for i in range(len(test_labels)):
        curr_correct = list(test_labels[i]) == predictions[i]
        if curr_correct:
            correct += 1
        if sum(test_labels[i]) > 1:
            num_multi_label_samples += 1
            if curr_correct:
                num_multi_label_correct += 1

    # Calculate metrics.
    if num_multi_label_samples == 0:
        num_multi_label_samples += 1
    accuracy = correct / len(test_labels)
    chunk_accuracy = chunk_correct / num_chunks
    frac_multi_label = num_multi_label_samples / len(test_labels)
    single_label_accuracy = (correct - num_multi_label_correct) / (len(test_labels) - num_multi_label_samples)
    multi_label_accuracy = num_multi_label_correct / num_multi_label_samples

    # Calculate label count target/prediction matrix.
    stsp = 0  # Single target, single prediction
    stmp = 0  # Single target, multi prediction
    mtsp = 0  # Multi target, single prediction
    mtmp = 0  # Multi target, multi prediction
    for i in range(len(test_labels)):
        if sum(test_labels[i]) > 1:
            if sum(predictions[i]) > 1:
                mtmp += 1
            else:
                mtsp += 1
        else:
            if sum(predictions[i]) > 1:
                stmp += 1
            else:
                stsp += 1

    assert stsp + stmp + mtsp + mtmp == len(test_labels)

    # Print results.
    print(" Done!")
    print("Accuracy: " + str(accuracy))
    print("Chunk Accuracy: " + str(chunk_accuracy))
    print("Single-Label target Accuracy: " + str(single_label_accuracy))
    print("Multi-Label target Accuracy: " + str(multi_label_accuracy))
    print("Fraction multi-label: " + str(frac_multi_label))
    print("Fraction multi-label prediction: ")

    print("Multi-label true - Single-label prediction: ", mtsp)
    print("Multi-label true - Multi-label prediction: ", mtmp)
    print("Single-label true - Single-label prediction: ", stsp)
    print("Single-label true - Multi-label prediction: ", stmp)

    # TODO: Confusion matrix, other stats, save results to file...
    # TODO: n-label confusion matrix!
    # Chunk accuracy
    # One label accuracy
    # More than one label accuracy
