
"""
This code generates audio wav file given the text representing the audio file from our vocab.
"""
import soundfile as sf
import librosa
import numpy as np


def generate_wav_from_tokens(tokens, id):
    audio_folder=  "./samples/cluster"
    output_path    = f"../generated/{id}_synthesised.wav"
    # silent_path      = "./samples/silent.wav"


    # silent_data, _ = sf.read(silent_path)
    # if silent_data.ndim > 1:
    #     silent_data = np.mean(silent_data, axis=1)

    chosen_paths =[]

    for token in tokens:
        list_of_sub_tokens = []
        
        for i in range(len(token)):
            list_of_sub_tokens.append(token[i])

        for sub_token in list_of_sub_tokens:
            cluster_number = sub_token
            path_of_folder_to_pick = audio_folder+f"{cluster_number}.wav"
            chosen_paths.append(path_of_folder_to_pick)

    print("[*] Chosing samples finished")
    first_data, target_sr = sf.read(chosen_paths[0])
    if first_data.ndim > 1:
        first_data = np.mean(first_data, axis=1)

    segments = [first_data]
    segments = []
    for i, path in enumerate(chosen_paths):
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        segments.append(data)

    combined = np.concatenate(segments)
    sf.write(output_path, combined, samplerate=48000)