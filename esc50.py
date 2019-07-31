import file_utils
import csv
import os

csv_indexing_file = 'ESC-50-master/meta/esc50.csv'
audio_dir = 'ESC-50-master/audio'

def get_training_data(label):
    num_files = 0
    dir, data_exists = file_utils.get_training_data_dir(label)
    if(data_exists):
        print(f'Training data for {label} already exists')
        return

    # creating directory
    os.mkdir(dir)
    # iterate through csv file
    # if a breath file index is found, copy the index's wav file to breath folder
    file = open(csv_indexing_file, "rU")
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        audio_classificaton = row[3]
        if(audio_classificaton == label):
            # file name is stored in the first column
            og_file_loc = f'{audio_dir}/{row[0]}'
            copy_file_loc = f'{dir}/{label}_{num_files}.wav'
            file_utils.copy_to(og_file_loc, copy_file_loc)
            num_files += 1


if __name__ == '__main__':
    get_training_data('breathing')
