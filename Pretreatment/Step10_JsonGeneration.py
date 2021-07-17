import os
import json
import tqdm
import numpy
from Tools import FoldSearcher

if __name__ == '__main__':
    transcript_path = 'D:/PythonProject/IEMOCAP_Data/Step9_TranscriptTreatment/'
    audio_path = 'D:/PythonProject/IEMOCAP_Data/Step4_AudioNormalization/'
    video_path = 'D:/PythonProject/IEMOCAP_Data/Step8_VideoNormalization/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/IEMOCAP_DATA/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    file_list = FoldSearcher(audio_path)
    total_sample = []

    current_part = ''
    for file_path in tqdm.tqdm(file_list):
        file_name = file_path[-file_path[::-1].find('\\'):-4]

        current_sample = {}
        if file_path.find('improve') != -1:
            current_sample['Part'] = 'improve'
        else:
            current_sample['Part'] = 'script'
        if file_path.find('Female') != -1:
            current_sample['Gender'] = 'Female'
        else:
            current_sample['Gender'] = 'Male'
        current_sample['Session'] = file_path[file_path.find('Session'):file_path.find('Session') + 8]
        current_sample['FileName'] = file_name

        if file_path.find('ang') != -1: current_sample['Emotion'] = 'ang'
        if file_path.find('exc') != -1: current_sample['Emotion'] = 'hap'
        if file_path.find('hap') != -1: current_sample['Emotion'] = 'hap'
        if file_path.find('neu') != -1: current_sample['Emotion'] = 'neu'
        if file_path.find('sad') != -1: current_sample['Emotion'] = 'sad'

        with open(file_path.replace('Step4_AudioNormalization', 'Step9_TranscriptTreatment').replace('csv', 'txt'),
                  'r') as file:
            transcript_data = file.readlines()[0]
            transcript_data = transcript_data.lower()
        current_sample['Text'] = transcript_data

        # audio_data = numpy.genfromtxt(fname=file_path, dtype=float, delimiter=',')
        # current_sample['Audio'] = audio_data.tolist()
        video_data = numpy.genfromtxt(fname=file_path.replace('Step4_AudioNormalization', 'Step8_VideoNormalization'),
                                      dtype=float, delimiter=',')
        current_sample['Video'] = video_data.tolist()

        if current_part == '':
            current_part = current_sample['Part'] + '_' + current_sample['Gender'] + '_' + current_sample['Session']
        if current_part != current_sample['Part'] + '_' + current_sample['Gender'] + '_' + current_sample['Session']:
            json.dump(total_sample, open(os.path.join(save_path, 'IEMOCAP_Video_%s.json' % current_part), 'w'))
            total_sample = []
            current_part = current_sample['Part'] + '_' + current_sample['Gender'] + '_' + current_sample['Session']

        total_sample.append(current_sample)
        # break

    json.dump(total_sample, open(os.path.join(save_path, 'IEMOCAP_Video_%s.json' % current_part), 'w'))
