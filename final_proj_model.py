from model import PopMusicTransformer
import numpy as np
import pdb

class MajMinPopMusicTransformer(PopMusicTransformer):
    def __init__(self, checkpoint, is_training=False):
        PopMusicTransformer.__init__(self, checkpoint, is_training)

    def get_six_seven_indices(self, all_events):
        """
        """
        return []

    def convert_events_to_words(self, all_events):
        """event to word
        """
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        print('OOV event found: {}'.format(e))
            all_words.append(words)

        return all_words

    def prepare_mtom_data(self, midi_paths):
        """
        """
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)

        # get indices of sixths and sevenths
        six_seven_indices = get_six_seven_indices(all_events)

        all_words = self.convert_events_to_words(all_events)

        segments = []
        for words in all_words:
            song = []
            for i in range(0, len(words), self.x_len):
                x = words[i:i+self.x_len]
                x = np.array(x)
                song.append(x)
            song = np.array(song)
            segments.append(song)
        segments = np.array(segments)

        return segments, six_seven_indices

    def evaluate_mtom_67s(self, mtom_data):
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
