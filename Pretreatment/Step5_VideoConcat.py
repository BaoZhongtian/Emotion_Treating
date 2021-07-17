import os
import tqdm
import numpy
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step2_SelectIntersection/Video-%s/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step5_VideoConcat/'

    file_list = FoldSearcher(load_path % 'Hand')
    for file_path in tqdm.tqdm(file_list):
        with open(file_path, 'r') as file:
            data_hand_raw = file.readlines()
        data_hand = []
        for indexX in range(2, len(data_hand_raw)):
            treat_line = data_hand_raw[indexX][0:-1].split(' ')[2:]
            data_hand.append([float(x) for x in treat_line])

        with open(file_path.replace('Hand', 'Head'), 'r') as file:
            data_head_raw = file.readlines()
        data_head = []
        for indexX in range(2, len(data_head_raw)):
            treat_line = data_head_raw[indexX][0:-1].split(' ')[2:]
            data_head.append([float(x) for x in treat_line])

        with open(file_path.replace('Hand', 'Rotated'), 'r') as file:
            data_rotated_raw = file.readlines()
        data_rotated = []
        for indexX in range(2, len(data_rotated_raw)):
            treat_line = data_rotated_raw[indexX][0:-1].split(' ')[2:]
            data_rotated.append([float(x) for x in treat_line])

        fold_path = file_path[0:-file_path[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path % 'Hand', save_path)):
            os.makedirs(fold_path.replace(load_path % 'Hand', save_path))
        data_concat = numpy.concatenate([data_hand, data_hand, data_rotated], axis=1)

        with open(file_path.replace(load_path % 'Hand', save_path).replace('txt', 'csv'), 'w') as file:
            for indexX in range(numpy.shape(data_concat)[0]):
                for indexY in range(numpy.shape(data_concat)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(data_concat[indexX][indexY]))
                file.write('\n')
        # exit()
