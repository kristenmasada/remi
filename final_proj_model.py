"""
"""

import pdb
import re
import sys

import numpy as np
import pretty_midi as pm

from model import PopMusicTransformer

class MajMinPopMusicTransformer(PopMusicTransformer):
    def __init__(self, checkpoint, is_training=False):
        PopMusicTransformer.__init__(self, checkpoint, is_training)

    def get_keys(self, midi_paths):
        """Get minor key for each MTOM song.
        """
        keys = []
        OCTAVE = 12
        RELATIVE_MINOR_ADJ = 3

        for s in midi_paths:
            parsed_s = pm.PrettyMIDI(s)
            if len(parsed_s.key_signature_changes) > 0:
                key_num = parsed_s.key_signature_changes[0].key_number
            else:
                print('pretty_midi could not find a key signature for {}'.format(s))
                sys.exit(1)

            # check if key_num corresponds to a minor key.
            # pretty_midi minor keys are between 12 and 23,
            # and need to be adjusted so that they are between
            # 0 and 11, inclusive.
            if key_num >= OCTAVE:
                key_num -= OCTAVE
            # otherwise, key is in major, and needs
            # to be converted to relative minor key.
            elif key_num < OCTAVE:
                key_num = (key_num - RELATIVE_MINOR_ADJ) % OCTAVE

            print('song:', s)
            print('key num:', key_num, 'key:', pm.key_number_to_key_name(key_num + OCTAVE), '\n')
            keys.append(key_num)

        return keys

    def get_six_seven_indices(self, all_events, keys):
        """
        NOTE: make sure that I am using the MIDI files that do not have the
        additional instrument parts!
        """
        six_seven_indices = []
        SIXTH_INT = 8
        SEVENTH_INT = 10
        OCTAVE = 12

        for key,events in zip(keys, all_events):
            song_six_seven_indices = []
            sixth_scale_degree = (key + SIXTH_INT) % OCTAVE
            seventh_scale_degree = (key + SEVENTH_INT) % OCTAVE
            for idx,e in enumerate(events):
                if (e.name == 'Note On'
                    and (e.value % OCTAVE == sixth_scale_degree
                         or e.value % OCTAVE == seventh_scale_degree)):
                    pdb.set_trace()
                    song_six_seven_indices.append(idx)
            six_seven_indices.append(song_six_seven_indices)

        return six_seven_indices

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
                        sys.exit(1)
            all_words.append(words)

        return all_words

    def prepare_mtom_data(self, midi_paths):
        """
        """
        keys = self.get_keys(midi_paths)

        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)

        six_seven_indices = self.get_six_seven_indices(all_events, keys)

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
