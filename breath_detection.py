# breath signal peak amplitude is relative to speech signal peak amplitude
from audio_classificaton import feat_extract, cnn, nn


def find_breaths():
    return {}



def remove_breaths(infile, outfile):
    return


# Gets .wav files of breaths from the EC-50 database and copies and stores the
# files in training_data correct directory format
def get_training_data():
    num_breathing_files = 0
    breathing_dir = 'training_data/breathing'

    # iterate through csv file
    # if a breath file index is found, copy the index's wav file to breath folder
    file = open('ESC-50-master/meta/esc50.csv', "rU")
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        audio_classificaton = row[3]

        if(audio_classificaton == 'breathing'):
            # file name is stored in the first column
            og_file_loc = f'ESC-50-master/audio/{row[0]}'
            copy_file_loc = f'{breathing_dir}/breath_{num_breathing_files}.wav'
            file_utils.copy_to(og_file_loc, copy_file_loc)
            num_breathing_files += 1



if __name__ == '__main__':
    model = file_utils.get_classification_model('breathing')
    if(model == None):
        # Check if features have been extracted
        feature_file, label_file = file_utils.get_feature_files('breathing')
        if(feature_file == None or label_file == None):
            # Check if training data has been retrieved
            if(file_utils.get_training_data_dir('breathing') == None):
                get_training_data()
            # Training Data will be present. Extract Feutures
            extract_features()
        # Now that feature files will be present, train the model
        train()
    # Now the model will be trained.
