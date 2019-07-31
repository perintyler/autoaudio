import wave
# https://docs.python.org/3/library/wave.html
class Audio:
    def __init__(self, file):
        if file.endswith('.wav'):
            self.file_ext = '.wav'
            with wave.open(file, "rb") as wave_file:
                self.props = wave_file.getparams()
        else:
            raise Exception(f'File type not supported for {file}')


    # # creates an audio object from a byte string
    # def __init__(self, raw_data, file_ext, sample_rate):
    #     pass

    def write_to_file(self, loc):
        return

    def replace_with_silence(start_time, end_time):
        return
