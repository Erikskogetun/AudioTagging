from data_synthesis import *
import subprocess

# ../../sox-14.4.2/sox "C:/Users/Erik Skogetun/Desktop/Skola KTH/DT2119 - Speech Recognition/Project/Training-files/2c1c3fe0.wav" tmp_1.wav norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse

# "../../sox-14.4.2/sox" "C:/Users/Erik Skogetun/Desktop/Skola KTH/DT2119 - Speech Recognition/Project/Training-files/4c9da128.wav" tmp_1.wav norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse

rdp = "../../train_curated/"
output_path = "output/"
chunk_size=128
test_frac=0.2
remove_silence=True
n_mels=64
generate_mixes=True
mix_order=2
debug_skip=True

generate_data(rdp, output_path, chunk_size, test_frac, remove_silence, n_mels, generate_mixes, mix_order, debug_skip)



'''

def _remove_silence(file_path, aug_audio_file):
    aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"

    subprocess.call('"../../sox-14.4.2/sox\"' + " \"" +  file_path  + "\" " + aug_audio_file + " " + aug_cmd)

    assert os.path.exists(aug_audio_file), "SOX Problem ... clipped wav does not exist! (while removing silence)"


# TODO: this may cause clipping as files were normalized to -0.1 in silence removal stage.
# TODO: Jim or someone: listen to some of the generated mixes and make sure there isn't clipping present when two or
# more sounds overlap.
def _sum_audio(audio_files, aug_audio_file):
    cmd = '"../../sox-14.4.2/sox\" -m '
    for f in audio_files:
        cmd = cmd + f + " "
    cmd = cmd + aug_audio_file

    subprocess.call(cmd)

    assert os.path.exists(aug_audio_file), "SOX Problem ... clipped wav does not exist! (while summing audio)"

'''
