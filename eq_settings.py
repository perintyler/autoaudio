# https://github.com/carlthome/python-audio-effects/blob/master/pysndfx/dsp.py

lowShelf = {
    'gain': -20.0,
    'frequency': 80, # frequency center
    'slope': 1
}

highpass = {
    'frequency': 100,
    'q': 0.63
}

lowFrequency = {
    'frequency': 150, #frequency center
    'db': 2.0, # gain
    'q': 1.72
}

# frequency center (hz), gain (db), Q
midFrequency = {
    'frequency': 350,
    'db': -3.0,
    'q': 1.9
}

highShelf = {
    'gain': 1, # db
    'frequency': 2000, # hz
    'slope': 0.5
}
