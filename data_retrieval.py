import file_utils
import csv

def get_esc50_training_data():
    breath_file_count = 0
    cough_file_count = 0
    data_path = 'ESC-50-master/audio/'
    # iterate through csv file
    # if a breath file index is found, copy the index's wav file to breath folder
    file = open('ESC-50-master/meta/esc50.csv', "rU")
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        audio_classificaton = row[3]
        # file name is stored in the first column
        file_loc = f'ESC-50-master/audio/{row[0]}'
        if(audio_classificaton == 'breathing'):
            file_utils.copy_to(file_loc, f'breath/esc-{breath_file_count}.wav')
            breath_file_count += 1
        elif(audio_classificaton == 'coughing'):
            file_utils.copy_to(file_loc, f'cough/esc-{cough_file_count}.wav')
            cough_file_count += 1

            
if __name__ == '__main__':
    get_esc50_training_data()
