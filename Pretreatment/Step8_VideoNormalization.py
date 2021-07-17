import os
import tqdm
import numpy
from Tools import FoldSearcher
from sklearn.preprocessing import scale

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step7_VideoNanRemove/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step8_VideoNormalization/'

    fold_list = FoldSearcher(load_path)
    total_data = []

    for file_path in tqdm.tqdm(fold_list):
        file_data = numpy.genfromtxt(file_path, dtype=float, delimiter=',')
        total_data.extend(file_data)

    print(numpy.shape(total_data))
    total_data = scale(total_data)

    start_position = 0
    for file_path in tqdm.tqdm(fold_list):
        file_data = numpy.genfromtxt(file_path, dtype=float, delimiter=',')
        fold_path = file_path[0:-file_path[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path, save_path)):
            os.makedirs(fold_path.replace(load_path, save_path))

        with open(file_path.replace(load_path, save_path), 'w') as file:
            batch_data = total_data[start_position:start_position + numpy.shape(file_data)[0]]
            for indexX in range(numpy.shape(batch_data)[0]):
                for indexY in range(numpy.shape(batch_data)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(batch_data[indexX][indexY]))
                file.write('\n')
        start_position += numpy.shape(file_data)[0]
