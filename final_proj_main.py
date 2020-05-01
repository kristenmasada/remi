"""
    Author: Kristen Masada
    Email: km942412@ohio.edu

    Description: This main program initializes the Pop Music
    Transformer model, converts the Mtom songs to the input
    representation, and evaluates the model on the sixths and
    sevenths in Mtom.

    Date: April 30, 2020
"""


from final_proj_model import MajMinPopMusicTransformer
from glob import glob
import os
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    model = MajMinPopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=False)

    mtom_midi_paths = glob('mtom_data/evaluation/w_67_instr/*.mid')
    mtom_data, \
    six_seven_indices, \
    six_seven_labels = model.prepare_mtom_data(midi_paths=mtom_midi_paths, ticks=120)

    six_seven_preds = model.evaluate_mtom_67s(mtom_data, six_seven_indices)
    model.compute_mtom_67_acc(six_seven_preds, six_seven_labels, mtom_midi_paths)
    
    model.close()

if __name__ == '__main__':
    main()
