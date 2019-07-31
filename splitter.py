# Split an audio file into multiple audio file each consiting of an
# individual attack



# The attack starts when amplitude starts increasing. Once a decrease of
# amplitude starts, the file should end whenever the next increase happens
# there should be a slope threshold that has to be reached to institue
# amplitude increase. There should be a min file time.
def find_attacks():
    return []


# If amplitude does not reach 0 (or baseline amplitude before attack) after a
# attack starts, then it is not a complete 'note'


# It would be cool to be able to tell if an audio clip contains isolate 'notes'



# Using attack and the initial amplitude slope decrease, an amplitude regression
# function should be able to be made which I'm pretty sure is logarthmic
# If the full sound of every attack of every audio source can be predicted,
# it should be possible to seperate all sound sources into seperate audio files
class Frame:
    def __init__(self):
        return


class Note:
    def __init__(self):
        self.startTime = 'tbt'

    def find_end(self, file):
        return

# A 'note' has an attack, a hold time, and then a slow death time
# The hold time will probably be the hardest caviat in this idea
