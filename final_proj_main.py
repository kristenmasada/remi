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
    #pdb.set_trace()
    mtom_data, \
    six_seven_indices = model.prepare_mtom_data(midi_paths=mtom_midi_paths, ticks=120)
    model.evaluate_mtom_67s(mtom_data, six_seven_indices)

    model.close()

if __name__ == '__main__':
    main()
