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

        note_items = utils.quantize_items(note_items, ticks)

        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)

        if normal_six_seven_note_items:
            normal_six_seven_note_items = utils.quantize_items(normal_six_seven_note_items, ticks)
            normal_max_time = normal_six_seven_note_items[-1].end
            normal_six_seven_groups = utils.group_items(normal_six_seven_note_items, normal_max_time)
            normal_events = utils.item2event(normal_six_seven_groups)

        else:
            normal_events = []

        if raised_six_seven_note_items:
            raised_six_seven_note_items = utils.quantize_items(raised_six_seven_note_items, ticks)
            raised_max_time = raised_six_seven_note_items[-1].end
            raised_six_seven_groups = utils.group_items(normal_six_seven_note_items, raised_max_time)
            raised_events = utils.item2event(raised_six_seven_groups)
        else:
            raised_events = []

        return events, normal_events, raised_events

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

    def get_six_seven_indices(self, all_events, all_normal_six_seven_events,
                              all_raised_six_seven_events, midi_paths):
        """
        NOTE: make sure that I am using the MIDI files that do not have the
        additional instrument parts!
        """
        pdb.set_trace()
        """
        six_seven_indices = []
        SIXTH_INT = 8
        SEVENTH_INT = 10
        OCTAVE = 12
        song_idx = 0

        for key,events in zip(keys, all_events):
            song_six_seven_indices = []
            sixth_scale_degree = (key + SIXTH_INT) % OCTAVE
            seventh_scale_degree = (key + SEVENTH_INT) % OCTAVE
            for idx,e in enumerate(events):
                if (e.name == 'Note On'
                    and (e.value % OCTAVE == sixth_scale_degree
                         or e.value % OCTAVE == seventh_scale_degree)):
                    #print('note:', pm.note_number_to_name(e.value))
                    song_six_seven_indices.append(idx)
            six_seven_indices.append(song_six_seven_indices)

            print('song', midi_paths[song_idx], '# 6/7 indices:', len(song_six_seven_indices))
            song_idx += 1
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
                        sys.exit(1)
            all_words.append(words)

        return all_words

    def prepare_mtom_data(self, midi_paths, ticks):
        """
        """
        #keys = self.get_keys(midi_paths)

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
