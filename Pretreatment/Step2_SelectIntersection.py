import os
import shutil
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step1_SelectFromSource/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step2_SelectIntersection/'

    audio_files = FoldSearcher(os.path.join(load_path, 'Audio'))
    hand_files = FoldSearcher(os.path.join(load_path, 'Video-Hand'))
    head_files = FoldSearcher(os.path.join(load_path, 'Video-Head'))
    rotated_files = FoldSearcher(os.path.join(load_path, 'Video-Rotated'))
    transcript_files = FoldSearcher(os.path.join(load_path, 'Transcript'))

    total_counter = 0
    for filepath in audio_files:
        if filepath.replace('Audio', 'Video-Hand').replace('wav', 'txt') not in hand_files: continue
        if filepath.replace('Audio', 'Video-Head').replace('wav', 'txt') not in head_files: continue
        if filepath.replace('Audio', 'Video-Rotated').replace('wav', 'txt') not in rotated_files: continue
        if filepath.replace('Audio', 'Transcript').replace('wav', 'txt') not in transcript_files: continue
        total_counter += 1

        fold_path = filepath[0:-filepath[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path, save_path)):
            os.makedirs(fold_path.replace(load_path, save_path))
            os.makedirs(fold_path.replace(load_path, save_path).replace('Audio', 'Video-Hand'))
            os.makedirs(fold_path.replace(load_path, save_path).replace('Audio', 'Video-Head'))
            os.makedirs(fold_path.replace(load_path, save_path).replace('Audio', 'Video-Rotated'))
            os.makedirs(fold_path.replace(load_path, save_path).replace('Audio', 'Transcript'))
        shutil.copy(filepath, filepath.replace(load_path, save_path))
        shutil.copy(filepath.replace('Audio', 'Video-Hand').replace('wav', 'txt'),
                    filepath.replace(load_path, save_path).replace('Audio', 'Video-Hand').replace('wav', 'txt'))
        shutil.copy(filepath.replace('Audio', 'Video-Head').replace('wav', 'txt'),
                    filepath.replace(load_path, save_path).replace('Audio', 'Video-Head').replace('wav', 'txt'))
        shutil.copy(filepath.replace('Audio', 'Video-Rotated').replace('wav', 'txt'),
                    filepath.replace(load_path, save_path).replace('Audio', 'Video-Rotated').replace('wav', 'txt'))
        shutil.copy(filepath.replace('Audio', 'Transcript').replace('wav', 'txt'),
                    filepath.replace(load_path, save_path).replace('Audio', 'Transcript').replace('wav', 'txt'))

    print(total_counter)
