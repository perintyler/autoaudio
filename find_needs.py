# This file will detect what the audio needs

# Types of categories for need accessment:
#   - EQing:
#       + Need to search for harsh frequencies
#   - BG Noise reduction:
#       + Maybe like a speech percentage detector
#       + Enviromental sounds detection
#   - Reverb/echo amount
#       + I think there is room size estimation?
#       + calculating echo time?
#
#
#   - For bad audio quality, there should be a unique process for each category.


needs_dereverb = False
needs_bg_noise_removal = False
