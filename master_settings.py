# Compand (https://forum.doom9.org/showthread.php?t=165807)
# speech: 0.02,0.20 5:-60,-40,-10 -5 -90 0.1
# voice/radio:  0.01,1 -90,-90,-70,-70,-60,-20,0,0 -5
#
# podcast: 0.3,1 6:-70,-60,-20 -5 -90

# compand attack1,decay1{,attack2,decay2}
#[soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
#[gain [initial-volume-dB [delay]]]

inf = float('inf')
ninf = float('-inf')

# play infile compand .1,.2 −inf,−50.1,−inf,−50,−50 0 −90 .1
low_gate = {
    'attack': 0.1,
    'decay': 0.2,
    'soft_knee': ninf,
}

# play infile compand .1,.1 −45.1,−45,−inf,0,−inf 45 −90 .1

compand = {
    'attack': 0.3,
    'decay': 1,
    'soft_knee': 6.0,
    'threshold': -70,
    'db_from':-5.0,
    'db_to': -20.0
}

reverb = {
   'reverberance': 15,
   'hf_damping': 10,
   'room_scale': 20,
   'stereo_depth': 20,
   'pre_delay': 5,
   'wet_gain': 0,
   'wet_only': False
}
