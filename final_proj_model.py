"""
"""

import pdb
import re
import sys

import miditoolkit
import numpy as np
import pretty_midi as pm

from model import PopMusicTransformer
from utils import Item
import utils

class MajMinPopMusicTransformer(PopMusicTransformer):
    def __init__(self, checkpoint, is_training=False):
        PopMusicTransformer.__init__(self, checkpoint, is_training)

        self.DEFAULT_RESOLUTION = 480

    def get_mtom_notes(self, instruments):
        """
        """

        # remove marker notes
        D7 = 98
        C7 = 96
        D1 = 26
        C1 = 24

        for instr in instruments:
            for n in instr.notes[:]:
                if n.start == 0.0 and n.pitch == C7:
                    instr.name = "raised"
                    instr.notes.remove(n)
                elif n.start == 0.0 and n.pitch == C1:
                    instr.name = "raised"
                    instr.notes.remove(n)
                elif n.start == 0.0 and n.pitch == D7:
                    instr.name = "normal"
                    instr.notes.remove(n)
                elif n.start == 0.0 and n.pitch == D1:
                    instr.name = "normal"
                    instr.notes.remove(n)
                elif n.start > 0.0:
                    break

        normal_six_seven_notes = []
        raised_six_seven_notes = []
        notes = []
        for instr in instruments:
            if instr.name == "normal":
                normal_six_seven_notes += instr.notes
            elif instr.name == "raised":
                raised_six_seven_notes += instr.notes
            else:
                notes += instr.notes

        return normal_six_seven_notes, raised_six_seven_notes, notes

    def process_notes(self, notes):
        """
        """
        note_items = []
        notes.sort(key=lambda x: (x.start, x.pitch))
        for note in notes:
            note_items.append(Item(
                name='Note',
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch))
        note_items.sort(key=lambda x: x.start)

        return note_items

    def process_tempos(self, midi_obj):
        """
        """
        # tempo
        tempo_items = []
        for tempo in midi_obj.tempo_changes:
            tempo_items.append(Item(
                name='Tempo',
                start=tempo.time,
                end=None,
                velocity=None,
                pitch=int(tempo.tempo)))
        tempo_items.sort(key=lambda x: x.start)
        # expand to all beat
        max_tick = tempo_items[-1].start
        existing_ticks = {item.start: item.pitch for item in tempo_items}
        wanted_ticks = np.arange(0, max_tick+1, self.DEFAULT_RESOLUTION)
        output = []
        for tick in wanted_ticks:
            if tick in existing_ticks:
                output.append(Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick]))
            else:
                output.append(Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch))
        tempo_items = output

        return tempo_items

    def get_six_seven_note_indices(self, note_items, six_seven_note_items):
        """
        """
        six_seven_indices = []
        for n_idx,n in enumerate(note_items):
            for s_n in six_seven_note_items:
                if (n.start == s_n.start
                    and n.pitch == s_n.pitch):
                    six_seven_indices.append(n_idx)
        return six_seven_indices

    def mtom_read_items(self, file_path):
        """
        """
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)

        normal_six_seven_notes, \
        raised_six_seven_notes, \
        notes = self.get_mtom_notes(midi_obj.instruments)

        note_items = self.process_notes(notes)
        normal_six_seven_note_items = self.process_notes(normal_six_seven_notes)
        raised_six_seven_note_items = self.process_notes(raised_six_seven_notes)

        tempo_items = self.process_tempos(midi_obj)

        return (note_items,
                normal_six_seven_note_items,
                raised_six_seven_note_items,
                tempo_items)

    def mtom_extract_events(self, input_path, ticks=120):
        """
        """
        note_items, \
        normal_six_seven_note_items, \
        raised_six_seven_note_items, \
        tempo_items = self.mtom_read_items(input_path)

        normal_six_seven_note_indices = self.get_six_seven_note_indices(note_items, normal_six_seven_note_items)
        raised_six_seven_note_indices = self.get_six_seven_note_indices(note_items, raised_six_seven_note_items)

        note_items = utils.quantize_items(note_items, ticks)

        normal_six_seven_note_items = [note_items[idx] for idx in normal_six_seven_note_indices]
        raised_six_seven_note_items = [note_items[idx] for idx in raised_six_seven_note_indices]

        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)

        return events, normal_six_seven_note_items, raised_six_seven_note_items

    def get_six_seven_indices(self, all_events, all_normal_six_seven_notes,
                              all_raised_six_seven_notes, midi_paths):
        """
        NOTE: # of 6/7's found in EKNM Solo is wrong. Come back to this later and fix it!
        """
        six_seven_indices = []
        for song_idx,events in enumerate(all_events):
            song_six_seven_notes = all_normal_six_seven_notes[song_idx] + all_raised_six_seven_notes[song_idx]
            song_six_seven_notes.sort(key=lambda x: x.start)
            song_six_seven_indices = []
            for event_idx,event in enumerate(events):
                if event.name == 'Note On':
                    for six_seven in song_six_seven_notes[:]:
                        if (event.value == six_seven.pitch
                            and event.time == six_seven.start):
                            song_six_seven_notes.remove(six_seven)
                            song_six_seven_indices.append(event_idx)
                            break

            six_seven_indices.append(song_six_seven_indices)
            #print('song:', midi_paths[song_idx], 'num 67s found:', len(song_six_seven_indices))

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

    def prepare_mtom_data(self, midi_paths, ticks):
        """
        """

        # extract events
        all_events = []
        all_normal_six_seven_events = []
        all_raised_six_seven_events = []
        for path in midi_paths:
            events, \
            normal_six_seven_events, \
            raised_six_seven_events = self.mtom_extract_events(path, ticks)
            all_events.append(events)
            all_normal_six_seven_events.append(normal_six_seven_events)
            all_raised_six_seven_events.append(raised_six_seven_events)

        six_seven_indices = self.get_six_seven_indices(all_events,
                                                       all_normal_six_seven_events,
                                                       all_raised_six_seven_events,
                                                       midi_paths)

        all_words = self.convert_events_to_words(all_events)

        """
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
        """
        segments = np.array(all_words)

        return segments, six_seven_indices

    def get_six_seven_prediction(self, logits, ss_idx, song_data):
        """
        """
        normal_67_word = song_data[ss_idx]
        normal_67_event = self.word2event[normal_67_word]
        normal_67_pitch = int(normal_67_event.split("_")[-1])
        raised_67_pitch = normal_67_pitch + 1
        raised_67_event = "Note On_" + str(raised_67_pitch)
        raised_67_word = self.event2word[raised_67_event]

        normal_67_score = logits[normal_67_word]
        raised_67_score = logits[raised_67_word]

        if raised_67_score > normal_67_score:
            return "raised"
        elif raised_67_score <= normal_67_score:
            return "normal"

    def evaluate_mtom_67s(self, mtom_data, six_seven_indices):
        """
        """
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]

        for song_data,ss_indices in zip(mtom_data, six_seven_indices):
            for ss_idx in ss_indices:
                words = [song_data[:ss_idx]]

                words_length = len(words[0])
                temp_x = np.zeros((self.batch_size, words_length))
                for b in range(self.batch_size):
                    for i,w in enumerate(words[b]):
                        temp_x[b][i] = w

                feed_dict = {self.x: temp_x}
                for m, m_np in zip(self.mems_i, batch_m):
                    feed_dict[m] = m_np

                # model (predictiosn)
                _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
                _logit = _logits[-1, 0]
                ss_idx_event = self.word2event[song_data[ss_idx]]
                six_seven_pred = self.get_six_seven_prediction(_logit, ss_idx, song_data)
                print('six seven prediction:', six_seven_pred)

                pdb.set_trace()
