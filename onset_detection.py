from pyAudioAnalysis import audioFeatureExtraction



def energy_entropy(amplitudes, sr):
    features, f_names = audioFeatureExtraction.stFeatureExtraction(amplitudes, sr, 0.050*sr, 0.025*sr);
    return features[f_names.index('energy_entropy')]



# Use amplitude changes (convex hull) for potential onsets

# use energy of entropy frequency spikes (melspectrogram) to confirm  
