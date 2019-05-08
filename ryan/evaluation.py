import numpy as np
from keras.models import load_model


def evaluate(model_path, test_set_path, label_threshold):
    # Load test set
    test_chunks = np.load(test_set_path + 'test_chunks.npy')
    test_labels = np.load(test_set_path + 'test_labels.npy')

    # Load model.
    model = load_model(model_path)

    # For each set of chunks in test_chunks, make predictions for each of them.
    num_chunks = 0
    chunk_correct = 0
    predictions = []
    for sample_index, chunk_set in enumerate(test_chunks):
        # Tally the number of chunks we've seen.
        num_chunks += len(chunk_set)

        # Make a prediction for each of the chunks in chunk_set.
        predictions = model.predict(x=chunk_set)

        # Tally the number of chunks which match the correct label.
        for p in predictions:
            if p == test_labels[sample_index]:
                chunk_correct += 1

        # Generate the resulting label vector for all chunks in this chunk set.
        res_label_vector = [0] * test_labels.shape[1]
        for i in range(len(res_label_vector)):
            if max(predictions[:, i]) > label_threshold:
                res_label_vector[i] = 1
        predictions.append(res_label_vector)

        # Print a couple prediction, res_label_vector pairs so we can see what's happening.
        if sample_index < 20:
            print("Sample index " + str(sample_index))
            print(predictions)
            print(res_label_vector)

    assert predictions.shape == test_labels.shape

    # Compare resulting label vectors against target label vectors in test_labels.
    correct = 0
    num_multi_label_correct = 0
    num_multi_label_samples = 0
    for i in range(len(test_labels)):
        curr_correct = test_labels[i] == predictions[i]
        if curr_correct:
            correct += 1
        if sum(test_labels[i]) > 1:
            num_multi_label_samples += 1
            if curr_correct:
                num_multi_label_correct += 1

    # Calculate metrics.
    accuracy = correct / len(test_labels)
    chunk_accuracy = chunk_correct / num_chunks
    frac_multi_label = num_multi_label_samples / len(test_labels)
    single_label_accuracy = (correct - num_multi_label_correct) / (len(test_labels) - num_multi_label_samples)
    multi_label_accuracy = num_multi_label_correct / num_multi_label_samples

    # Print results.
    print("Accuracy: " + str(accuracy))
    print("Chunk Accuracy: " + str(chunk_accuracy))
    print("Single Label Accuracy: " + str(single_label_accuracy))
    print("Multi Label Accuracy: " + str(multi_label_accuracy))
    print("Fraction multi-label: " + str(frac_multi_label))

    # TODO: Confusion matrix, other stats, save results to file...
    # Chunk accuracy
    # One label accuracy
    # More than one label accuracy

