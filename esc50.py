from file_utils import copy_to
import csv
import os

csv_indexing_file = 'ESC-50-master/meta/esc50.csv'
audio_dir = 'ESC-50-master/audio'

all_sounds = [
    'car_horn',
    'laughing',
    'crickets',
    'pouring_water',
    'vacuum_cleaner',
    'insects',
    'hand_saw',
    'glass_breaking',
    'wind',
    'church_bells',
    'train', 'rain',
    'sea_waves',
    'dog',
    'hen',
    'cat',
    'keyboard_typing',
    'clock_alarm',
    'crying_baby',
    'breathing',
    'coughing',
    'clock_tick',
    'door_wood_knock',
    'sneezing',
    'siren',
    'chirping_birds',
    'crow',
    'frog',
    'cow',
    'helicopter',
    'thunderstorm',
    'airplane',
    'mouse_click',
    'footsteps',
    'crackling_fire',
    'door_wood_creaks',
    'snoring',
    'toilet_flush',
    'can_opening',
    'washing_machine',
    'water_drops',
    'chainsaw',
    'brushing_teeth',
    'fireworks',
    'rooster',
    'sheep',
    'drinking_sipping',
    'engine',
    'pig',
    'clapping'
]

def get_training_dir(sound):
    dir = f'{settings.training_data_dir}/{sound}/'
    # Check to see if the directory exists and is not empty
    training_data_exists = os.path.isdir(dir) and not is_dir_empty(dir)
    return dir, training_data_exists

def retrieve_training_data(sound):
    num_files = 0
    dir, data_exists = get_training_dir(sound)
    if data_exists:
        print(f'Training data for {sound} already exists')
        return
    elif not os.path.isdir(dir):
        os.mkdir(dir)

    file = open(csv_indexing_file, "rU")
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        audio_classificaton = row[3]
        if(audio_classificaton == sound):
            # file name is stored in the first column
            og_file_loc = f'{audio_dir}/{row[0]}'
            copy_file_loc = f'{dir}/{sound}_{num_files}.wav'
            copy_to(og_file_loc, copy_file_loc)
            num_files += 1

def retrieve_all_sounds():
    if not os.path.isdir(settings.training_data_dir):
        os.mkdir(settings.training_data_dir)
        
    for sound in all_sounds:
        retrieve_training_data(sound)


def print_all_sounds():
    sounds = set()
    file = open(csv_indexing_file, "rU")
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        sound_classification = row[3]
        sounds.add(sound_classification)
    print(sounds)


if __name__ == '__main__':
    retrieve_all_sounds()
