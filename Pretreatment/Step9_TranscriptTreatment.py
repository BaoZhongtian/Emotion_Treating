import os
import re
import tqdm
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step1_SelectFromSource/Transcript/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step9_TranscriptTreatment/'

    file_list = FoldSearcher(load_path)
    for file_path in tqdm.tqdm(file_list):
        fold_path = file_path[0:-file_path[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path, save_path)):
            os.makedirs(fold_path.replace(load_path, save_path))

        with open(file_path, 'r') as file:
            file_data = file.readlines()

        sentence = ''
        for word_sample in file_data[1:-1]:
            word_sample = word_sample[0:-1].split(' ')
            if word_sample[-1].find('<') != -1: continue
            sentence += word_sample[-1] + ' '
        sentence = re.sub('\\(.*?\\)', '', sentence)

        with open(os.path.join(file_path.replace(load_path, save_path)), 'w') as file:
            file.write(sentence)
        # exit()
