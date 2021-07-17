import os
import tqdm
import shutil

if __name__ == '__main__':
    selected_path = 'D:/PythonProject/IEMOCAP_Data/Step0_SelectTranscript/'
    load_path = 'D:/PythonProject/IEMOCAP_full_release/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step1_SelectFromSource/'

    for talk_part in os.listdir(selected_path):
        for gender_part in os.listdir(os.path.join(selected_path, talk_part)):
            for session_part in os.listdir(os.path.join(selected_path, talk_part, gender_part)):
                print('Treating', talk_part, gender_part, session_part)

                for emotion_part in os.listdir(os.path.join(selected_path, talk_part, gender_part, session_part)):
                    if not os.path.exists(
                            os.path.join(save_path, 'Audio', talk_part, gender_part, session_part, emotion_part)):
                        os.makedirs(
                            os.path.join(save_path, 'Audio', talk_part, gender_part, session_part,
                                         emotion_part))
                        os.makedirs(
                            os.path.join(save_path, 'Video-Hand', talk_part, gender_part, session_part,
                                         emotion_part))
                        os.makedirs(
                            os.path.join(save_path, 'Video-Head', talk_part, gender_part, session_part,
                                         emotion_part))
                        os.makedirs(
                            os.path.join(save_path, 'Video-Rotated', talk_part, gender_part, session_part,
                                         emotion_part))
                        os.makedirs(
                            os.path.join(save_path, 'Transcript', talk_part, gender_part, session_part,
                                         emotion_part))

                    for filename in os.listdir(
                            os.path.join(selected_path, talk_part, gender_part, session_part, emotion_part)):
                        filename = filename.replace('.txt', '')
                        file_fold = filename[0:-filename[::-1].find('_') - 1]

                        try:
                            shutil.copy(
                                os.path.join(load_path, session_part, 'sentences', 'wav', file_fold, filename + '.wav'),
                                os.path.join(save_path, 'Audio', talk_part, gender_part, session_part, emotion_part,
                                             filename + '.wav'))
                        except:
                            raise RuntimeWarning(
                                'Could Not Found' + os.path.join(load_path, session_part, 'sentences', 'wav', file_fold,
                                                                 filename + '.wav'))

                        try:
                            shutil.copy(
                                os.path.join(load_path, session_part, 'sentences', 'MOCAP_hand', file_fold,
                                             filename + '.txt'),
                                os.path.join(save_path, 'Video-Hand', talk_part, gender_part, session_part,
                                             emotion_part, filename + '.txt'))
                        except:
                            print(
                                'Could Not Found' + os.path.join(load_path, session_part, 'sentences', 'MOCAP_hand',
                                                                 file_fold, filename + '.txt'))

                        try:
                            shutil.copy(
                                os.path.join(load_path, session_part, 'sentences', 'MOCAP_head', file_fold,
                                             filename + '.txt'),
                                os.path.join(save_path, 'Video-Head', talk_part, gender_part, session_part,
                                             emotion_part, filename + '.txt'))
                        except:
                            print(
                                'Could Not Found' + os.path.join(load_path, session_part, 'sentences', 'MOCAP_head',
                                                                 file_fold, filename + '.txt'))

                        try:
                            shutil.copy(
                                os.path.join(load_path, session_part, 'sentences', 'MOCAP_rotated', file_fold,
                                             filename + '.txt'),
                                os.path.join(save_path, 'Video-Rotated', talk_part, gender_part, session_part,
                                             emotion_part, filename + '.txt'))
                        except:
                            print(
                                'Could Not Found' + os.path.join(load_path, session_part, 'sentences', 'MOCAP_rotated',
                                                                 file_fold, filename + '.txt'))

                        try:
                            shutil.copy(
                                os.path.join(load_path, session_part, 'sentences', 'ForcedAlignment', file_fold,
                                             filename + '.wdseg'),
                                os.path.join(save_path, 'Transcript', talk_part, gender_part, session_part,
                                             emotion_part, filename + '.txt'))
                        except:
                            print(
                                'Could Not Found' + os.path.join(load_path, session_part, 'ForcedAlignment',
                                                                 'MOCAP_rotated', file_fold, filename + '.wdseg'))

