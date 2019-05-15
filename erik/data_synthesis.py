import numpy as np
import csv
import os
import librosa
import random

import matplotlib.pyplot as plt

import subprocess

labels_list = ['Bark', 'Raindrop', 'Finger_snapping', 'Run', 'Whispering', 'Acoustic_guitar', 'Strum', 'Hi-hat', 'Bass_drum', 'Crowd', 'Cheering', 'Frying_(food)', 'Chewing_and_mastication', 'Fart', 'Bass_guitar', 'Knock', 'Motorcycle', 'Stream', 'Male_singing', 'Crackle', 'Sigh', 'Burping_and_eructation', 'Female_singing', 'Tap', 'Female_speech_and_woman_speaking', 'Accelerating_and_revving_and_vroom', 'Clapping', 'Accordion', 'Zipper_(clothing)', 'Bus', 'Meow', 'Waves_and_surf', 'Microwave_oven', 'Child_speech_and_kid_speaking', 'Buzz', 'Car_passing_by', 'Toilet_flush', 'Purr', 'Church_bell', 'Electric_guitar', 'Marimba_and_xylophone', 'Trickle_and_dribble', 'Traffic_noise_and_roadway_noise', 'Harmonica', 'Male_speech_and_man_speaking', 'Slam', 'Keys_jangling', 'Sink_(filling_or_washing)', 'Water_tap_and_faucet', 'Squeak', 'Cricket', 'Fill_(with_liquid)', 'Skateboard', 'Shatter', 'Drawer_open_or_close', 'Race_car_and_auto_racing', 'Cupboard_open_or_close', 'Computer_keyboard', 'Writing', 'Sneeze', 'Drip', 'Bicycle_bell', 'Applause', 'Printer', 'Gong', 'Glockenspiel', 'Screaming', 'Yell', 'Cutlery_and_silverware', 'Walk_and_footsteps', 'Mechanical_fan', 'Gasp', 'Gurgling', 'Chink_and_clink', 'Tick-tock', 'Chirp_and_tweet', 'Hiss', 'Dishes_and_pots_and_pans', 'Bathtub_(filling_or_washing)', 'Scissors']


def generate_data(raw_data_path,
                  output_path,
                  chunk_size=128,
                  test_frac=0.2,
                  remove_silence=False,
                  n_mels=64,
                  generate_mixes=False,
                  mix_order=2,
                  debug_skip=False):
    if debug_skip:
        print("DEBUG SKIP ENABLED. SKIPPING ORDER 1 DATA!")

    # Note: it is expected that train_curated.csv is parallel to raw_data_path
    # Get filenames to target vector map
    filenames_to_labels = _filenames_to_labels(raw_data_path + '../')
    print('Loaded labels of ' + str(len(filenames_to_labels.keys())) + ' files.')

    train_main_chunks = []
    train_extra_chunks = []
    train_main_labels = []
    train_extra_labels = []
    test_chunks = []
    test_labels = []
    file_count = 0
    train_set_files = []
    print('looking at files in ' + raw_data_path)

    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if not file.endswith('.wav'):
                continue
            if debug_skip:
                train_set_files.append(file)
                continue
            # Report which file we're on to stdout.
            file_count += 1
            if file_count % 25 == 0:
                print("\r", str(file_count) + ' ' + file, end="")
            file_path = os.path.join(root, file)

            # Get target label vector for this file.
            label = filenames_to_labels[file]
            assert label

            # Remove silence from audio file.
            aug_audio_file = "tmp_sil.wav"
            if remove_silence:
                _remove_silence(file_path, aug_audio_file)
                file_path = aug_audio_file

            # Generate spectrogram of this file.
            # TODO: take params to allow non default arguments of _filename_to_spec
            try:
                spectrogram = _filename_to_spec(file_path, n_mels=n_mels)
            except ValueError:
                print('Value error occurred with file ' + file_path +
                      '! Skipping this file. (filecount ' + str(file_count) + ')')
                continue

            # Remove temp file if it exists.
            if remove_silence:
                if os.path.exists(aug_audio_file):
                    os.remove(aug_audio_file)
                else:
                    print('Tried to remove ' + aug_audio_file + ' but doesn\'t exist!')

            # Chunk the spectrogram.
            chunks = _spectrogram_to_chunks(spectrogram, chunk_size)

            """
            Test set should not contain any chunks from files which have chunks in training set!!!
            We choose to save the test set differently than the training set.
            Training set rows each correspond to some chunk who's label is saved in same row of
            corresponding labels file. Just looking at the train_main_chunks and train_extra_labels,
            it is ambiguous which chunks come from the same file.
            Because we want to make test set predictions by summing the output of the model for all chunks
            of a particular file, we can't save the test set in this way and instead save a list of lists of chunks.
            """
            # Handle chunks depending on if this file will be part of training or test set.
            if file_count < int((1 - test_frac) * len(files)):
                # This file is part of training.
                train_set_files.append(file)

                # Append first chunk to main_input data, and rest to extended_input data.
                train_main_chunks.append(chunks[0])
                train_main_labels.append(label)

                for i in range(1, len(chunks)):
                    train_extra_chunks.append(chunks[i])
                    train_extra_labels.append(label)
            else:
                # This file is part of test set.
                test_chunks.append(chunks)
                test_labels.append(label)

    # Save train and test sets.
    print("Generated spectrograms for " + str(file_count) + " files. Saving...", end='')
    # Save files.
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(output_path + 'train_main_chunks', train_main_chunks)
    np.save(output_path + 'train_extra_chunks', train_extra_chunks)
    np.save(output_path + 'train_main_labels', train_main_labels)
    np.save(output_path + 'train_extra_labels', train_extra_labels)
    np.save(output_path + 'test_chunks', test_chunks)
    np.save(output_path + 'test_labels', test_labels)
    print("Done!")

    del train_main_chunks
    del train_extra_chunks
    del train_main_labels
    del train_extra_labels
    del test_chunks
    del test_labels
    if generate_mixes:
        _generate_mixes(train_set_files=train_set_files,
                        filenames_to_labels=filenames_to_labels,
                        input_path=raw_data_path,
                        output_path=output_path,
                        chunk_size=chunk_size,
                        remove_silence=remove_silence,
                        n_mels=n_mels,
                        mix_order=mix_order)


# TODO: make remove_silence optional.
def _generate_mixes(train_set_files,
                    filenames_to_labels,
                    input_path,
                    output_path,
                    chunk_size=128,
                    remove_silence=False,
                    n_mels=64,
                    mix_order=2):
    print("Generating mixes.")
    """
    Make and save mixes.
    """

    for n in range(2, mix_order + 1):
        mixes_chunks = []
        mixes_labels = []

        # n is number sounds in mix. n=2 denotes pairs, n=3 denotes triples, ect.
        for mix_number in range(len(train_set_files)):

            files_in_mix = random.sample(train_set_files, n)
            print("\r", 'Mix ' + str(mix_number) + ' of ' + str(len(train_set_files)) + '. Mixing ' + str(files_in_mix), end="")

            # Remove silence from each audio file
            # TODO: make this optional. Right now ence is ignored and silence is removed regardless
            tmp_files = ['tmp_' + str(x + 1) + '.wav' for x in range(len(files_in_mix))]
            for i, mix_file in enumerate(files_in_mix):
                # print("Silence files in call: ", mix_file, tmp_files[i])
                _remove_silence(input_path + mix_file, tmp_files[i], mix_number)

            # Sum them
            sum_file = 'tmp_sum.wav'
            _sum_audio(tmp_files, sum_file, mix_number)

            # Generate spectrogram
            # TODO: take params to allow non default arguments of _filename_to_spec
            try:
                spectrogram = _filename_to_spec(sum_file, n_mels=n_mels)
            except ValueError:
                print('Value error occurred with file ' + sum_file +
                      '! Skipping this file.')
                continue

            # Remove temp files.
            for f in tmp_files + [sum_file]:
                if os.path.exists(f):
                    os.remove(f)
                else:
                    print('Tried to remove ' + f + ' but doesn\'t exist!')

            # Chunk the spectrogram (and just take first chunk)
            mixed_chunk_array = _spectrogram_to_chunks(spectrogram, chunk_size)

            # Generate new label
            individual_labels = [filenames_to_labels[f] for f in files_in_mix]
            mixed_label = np.max(individual_labels, axis=0)
            if mix_number < 5:
                print("Individual labels: " + str(individual_labels))
                print("Generated mixed label: " + str(mixed_label))

            # Append current chunk and label to output.
            for mixed_chunk in mixed_chunk_array:
                mixes_chunks.append(mixed_chunk)
                mixes_labels.append(mixed_label)

        print("Saving mixes of order " + str(n) + "...")
        np.save(output_path + 'mixes_chunks_' + str(n), mixes_chunks)
        np.save(output_path + 'mixes_labels_' + str(n), mixes_labels)
        print("Done!")

def _remove_silence(file_path, aug_audio_file, mix_number = None):
    aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
    
    os.system("../../sox-14.4.2/src/sox %s %s %s" % (file_path, aug_audio_file, aug_cmd))
    #subprocess.call('"../../sox-14.4.2/src/sox\"' + " \"" +  file_path  + "\" " + aug_audio_file + " " + aug_cmd)

    # Use this only if you want to save for debugging or similar
    # subprocess.call('"../../sox-14.4.2/sox\"' + " \"" +  file_path  + "\" " + "sumfiles/" + str(mix_number) + aug_audio_file + " " + aug_cmd)

    assert os.path.exists(aug_audio_file), "SOX Problem ... clipped wav does not exist! (while removing silence)"


# TODO: this may cause clipping as files were normalized to -0.1 in silence removal stage.
# TODO: Jim or someone: listen to some of the generated mixes and make sure there isn't clipping present when two or
# more sounds overlap.
def _sum_audio(audio_files, aug_audio_file, mix_number):

    # Measure the length of all clips
    # For some reason, there is sometimes clips lacking the duration-property.
    lengths = np.zeros(len(audio_files))
    for idx, f in enumerate(audio_files):
        p = subprocess.Popen(['../../sox-14.4.2/src/sox', '--i', f], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")

        length = str(output).split(' samples')[0].split(" ")[-1]
        lengths[idx] = int(length)

    len_argmax = lengths.argmax()
    len_max = lengths.max()

    generalcmd = '../../sox-14.4.2/src/sox\ -G -m '
    for idx, f in enumerate(audio_files):
        multitimes = int(len_max/lengths[idx])
        concat_cmd = '../../sox-14.4.2/src/sox\ '

        if multitimes > 1:
            concat_cmd = '../../sox-14.4.2/src/sox\ '
            for i in range(0, multitimes):
                concat_cmd = concat_cmd + f + " "

            f = f.split(".wav")[0] + "_res.wav"
            concat_cmd = concat_cmd + f
            subprocess.call(concat_cmd)
            #os.system(concat_cmd)

        generalcmd = generalcmd + f + " "

    savecmd = generalcmd + "sumfiles/" + str(mix_number) + "sum.wav"
    tempcmd = generalcmd + aug_audio_file

    #os.system(tempcmd)
    subprocess.call(tempcmd)

    # Use this only if you want to save for debugging or similar
    # subprocess.call(savecmd)

    assert os.path.exists(aug_audio_file), "SOX Problem ... clipped wav does not exist! (while summing audio)"


def _spectrogram_to_chunks(spectrogram, chunk_size):
    # Input spectrogram should be n X f where n is time domain and f is size of filterbank
    chunks = []
    if spectrogram.shape[0] < chunk_size:
        # Spectrogram is smaller than one chunk size, pad it.
        padding = np.full((chunk_size - spectrogram.shape[0], spectrogram.shape[1]), np.min(spectrogram))
        chunks.append(np.concatenate((spectrogram, padding)))
    else:
        i = 0
        while spectrogram.shape[0] - i >= chunk_size:
            chunks.append(spectrogram[i:i + chunk_size])
            i += chunk_size
    return chunks


def _filename_to_spec(file_path, n_fft=1024, sr=44100, mono=True, log_spec=False, n_mels=64, hop_length=512, fmax=None):
    samples, sr = librosa.load(file_path, sr=sr, mono=mono)

    # Compute stft
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                        pad_mode='reflect')

    # Get only frequencies and ignore phases.
    stft = np.abs(stft)

    # Select our spectrogram weighting.
    if log_spec:
        stft = np.log10(stft + 1)
    else:
        freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
        stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

    # Apply mel filterbank.
    spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax).T

    # Visualize
    #plt.pcolormesh(spectrogram.T)
    #plt.show()

    assert spectrogram.shape[1] == n_mels
    return spectrogram


def _filenames_to_labels(raw_data_path):
    filenames_to_labels = {}
    with open(raw_data_path + 'train_curated.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            # ignore first row
            if row[0] == 'fname':
                continue
            # map filename to label vector corresponding to that file
            filename = row[0]
            label_vector = [0] * len(labels_list)
            curr_labels = row[1].split(',')
            for l in curr_labels:
                label_vector[labels_list.index(l)] = 1
            filenames_to_labels[filename] = label_vector

    return filenames_to_labels
