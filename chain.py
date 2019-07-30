from pysndfx import AudioEffectsChain
import eq_settings
import master_settings

class EffectsChain(AudioEffectsChain):

    def compress(self):
        return self

    def equalize(self):
        return self.highpass(**eq_settings.highpass)
                    .equalizer(**eq_settings.lowFrequency)
                    .equalizer(**eq_settings.midFrequency)
                    .highshelf(**eq_settings.highShelf)


    def low_gate(self, threshold=-50):
        command = f'compand .1,.2 −inf,{level - 0.1},−inf,{level},{level} 0 {threshold} .1'
        return self.custom(command)

    def high_gate(self, threshold=-50):
        command = f'compand .1,.1 {level - 0.1},{level},−inf,0,−inf 45 −90 .1'
        return self.custom(command)

    def noise_removal(self):
    #     self.command.append('equalizer')
    #     self.command.append(frequency)
    #     self.command.append(str(q) + 'q')
    #     self.command.append(db)
        return self


    # used for deessing
    def desibilize(self):
        return self

    # used to simulate the effects of a pop filter
    def depop(self):
        return self
