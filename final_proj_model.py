"""
    Author: Kristen Masada
    Email: km942412@ohio.edu

    Description: This file contains the implementation for the
    MajMinPopMusicTransformer class.

    Date: April 30, 2020
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
        """Initialize the MajMinPopMusicTransformer model, setting the default
        input dimension, memory length, number of encoder layers, and other
        hyperparameters.

        Args:
            checkpoint: the filepath for the pre-trained model.
            is_training: boolean indicating if model should be used in
                         train or evaluation mode.
        """
        PopMusicTransformer.__init__(self, checkpoint, is_training)

        # tick step-size; used to get tempo information in input.
        self.DEFAULT_RESOLUTION = 480

    def get_mtom_notes(self, instruments):
        """Get all of the note objects associated with a midi file. Also get all
        of the notes corresponding to the normal 6/7's and raised 6/7's.

        Args:
            instruments: individual instrument parts in a midi file that has
                         been parsed.

        Return Values:
            normal_six_seven_notes: list of normal 6/7 miditoolkit Note objects.
            raised_six_seven_notes: list of raised 6/7 miditoolkit Note objects.
            notes: list of miditoolkit Note objects for all notes in song.
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
        """Convert miditoolkit note objects to Item objects.

        Args:
            notes: list of miditoolkit Note objects.

        Return values:
            note_items: list of Item objects.
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
        """Get information about all tempo changes for one midi file. Convert
        these to item objects.

        Args:
            midi_obj: parsed miditoolkit object for one midi file.

        Return values:
            tempo_items: list of Item objects corresponding to tempo information.
        """
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
        """Find indices of notes in note_items coresponding to notes in
        six_seven_note_items.

        Args:
            note_items: list of all note items.
            six_seven_note_items: list of six/seven notes to find in note_items.

        Return values:
            six_seven_indices: list of indices of six_seven_note_items in
                               note_items.
        """
        six_seven_indices = []
        for n_idx,n in enumerate(note_items):
            for s_n in six_seven_note_items:
                if (n.start == s_n.start
                    and n.pitch == s_n.pitch):
                    six_seven_indices.append(n_idx)
        return six_seven_indices

    def mtom_read_items(self, file_path):
        """Get note and tempo items from a single midi file.

        Args:
            file_path: Path to current midi file.

        Return values:
            note_items: list of note items for notes in midi file.
            normal_six_seven_note_items: list of note items for normal 6/7 notes
                                         in midi file.
            raised_six_seven_note_items: list of note items for raised 6/7 notes
                                         in midi file.
            tempo_items: list of tempo items from midi file.
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
        """Parse midi file to extract events (e.g. Note On, Note Velocity, etc.).

        Args:
            input_path: path to current midi file.
            ticks: used to quantize timings of items in song.

        Return values:
            events: list of events for song.
            normal_six_seven_note_items: list of note items for normal 6/7's in
                                         song.
            raised_six_seven_note_items: list of note items for raised 6/7's in
                                         song.
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
        """Get indices of 6/7 notes occurring in list of all events in song.

        Args:
            all_events: list of all events in current song.
            all_normal_six_seven_notes: list of all normal 6/7 notes in song.
            all_raised_six_seven_notes: list of all raised 6/7 notes in song.
            midi_paths: list of song names.

        Return values:
            six_seven_indices: list of indices of each 6/7 note in list of all
                               song events.
            six_seven_labels: nested list of ground truth labels for each 6/7 in
                              each song.
        """
        six_seven_indices = []
        six_seven_labels = []
        for song_idx,events in enumerate(all_events):
            song_normal_six_seven_notes = all_normal_six_seven_notes[song_idx]
            song_raised_six_seven_notes = all_raised_six_seven_notes[song_idx]
            song_six_seven_indices = []
            song_six_seven_labels = []
            for event_idx,event in enumerate(events):
                if event.name == 'Note On':
                    for six_seven in song_normal_six_seven_notes[:]:
                        if (event.value == six_seven.pitch
                            and event.time == six_seven.start):
                            song_normal_six_seven_notes.remove(six_seven)
                            song_six_seven_indices.append(event_idx)
                            song_six_seven_labels.append("normal")
                            break
                    for six_seven in song_raised_six_seven_notes[:]:
                        if (event.value == six_seven.pitch
                            and event.time == six_seven.start):
                            song_raised_six_seven_notes.remove(six_seven)
                            song_six_seven_indices.append(event_idx)
                            song_six_seven_labels.append("raised")
                            break
            six_seven_indices.append(song_six_seven_indices)
            six_seven_labels.append(song_six_seven_labels)
            #print('song:', midi_paths[song_idx], 'num 67s found:', len(song_six_seven_indices))

        return six_seven_indices, six_seven_labels

    def convert_events_to_words(self, all_events):
        """Convert each event in each song to its corresponding word in the
        loaded dictionary of tokens.

        Args:
            all_events: nested list of events in each song.

        Return values:
            all_words: nested list of words in each song.
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
        """Parse all Mtom songs and convert each to a list of words to use as
        input to the Transformer.

        Args:
            midi_paths: list of filepaths for Mtom songs.
            ticks: used to quantize the events in each Mtom song.

        Return values:
            segments: numpy array of words for all songs. Used as input to
                      Transformer model.
            six_seven_indices: nested list of the location of each 6/7 word in
                               segments.
            six_seven_labels: nested list of ground truth labels for each 6/7 in
                              each song.
        """

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

        six_seven_indices, \
        six_seven_labels = self.get_six_seven_indices(all_events,
                                                      all_normal_six_seven_events,
                                                      all_raised_six_seven_events,
                                                      midi_paths)

        all_words = self.convert_events_to_words(all_events)

        segments = np.array(all_words)

        return segments, six_seven_indices, six_seven_labels

    def get_six_seven_prediction(self, logits, ss_idx, song_data):
        """Get the model's prediction for a specific 6/7 based on its computed
        logit scores.

        Args:
            logits: the outputted scores from the Transformer.
            ss_idx: index of current 6/7 note.
            song_data: the list of words in the current song.
        Return values:
            The model's prediction (either "normal" or "raised").
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
        """Get the pre-trained Transformer model's predictions for each 6/7 note.

        Args:
            mtom_data: nested list of the input representation for each Mtom song.
            six_seven_indices: index of each 6/7 in mtom_data.

        Return values:
            six_seven_preds: nested list of the model's prediction for each 6/7
                             in each song (either "normal" or "raised").
        """
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        six_seven_preds = []
        for song_data,ss_indices in zip(mtom_data, six_seven_indices):
            prev_ss_idx = 0
            song_six_seven_preds = []
            for ss_idx in ss_indices:
                words = song_data[prev_ss_idx:ss_idx]
                words_length = len(words)

                temp_x = np.zeros((self.batch_size, words_length))
                for b in range(self.batch_size):
                    for i,w in enumerate(words):
                        temp_x[b][i] = w

                feed_dict = {self.x: temp_x}
                for m, m_np in zip(self.mems_i, batch_m):
                    feed_dict[m] = m_np

                # model (predictiosn)
                _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
                _logit = _logits[-1, 0]
                ss_idx_event = self.word2event[song_data[ss_idx]]
                six_seven_pred = self.get_six_seven_prediction(_logit, ss_idx, song_data)
                song_six_seven_preds.append(six_seven_pred)
                #print('pred:', six_seven_pred)

                prev_ss_idx = ss_idx
                batch_m = _new_mem
            six_seven_preds.append(song_six_seven_preds)

        return six_seven_preds

    def compute_mtom_67_acc(self, six_seven_preds, six_seven_labels, midi_paths):
        """Compute how many 6/7's the model predicts correctly per song and
        overall.

        Args:
            six_seven_preds: nested list of model's prediction for each 6/7 in
                             each song. Value for each 6/7 is either "normal" or
                             "raised".
            six_seven_labels: nested list of ground truth labels for each 6/7 in
                              each song.
            midi_paths: list of names of the midi files being evaluated on.
        """
        song_idx = 0
        correct = 0
        total = 0
        for song_preds,song_labels in zip(six_seven_preds, six_seven_labels):
            song_correct = 0
            song_total = len(song_labels)
            for p,l in zip(song_preds, song_labels):
                if p == l:
                    song_correct += 1
            song_acc = song_correct / song_total
            print('song:', midi_paths[song_idx])
            print('acc: {:.2f}%'.format(song_acc * 100.0))
            correct += song_correct
            total += song_total
            song_idx += 1
        overall_acc = correct / total
        print('overall acc: {:.2f}%'.format(overall_acc * 100.0))
