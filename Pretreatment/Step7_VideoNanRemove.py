import os
import tqdm
import numpy
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step5_VideoConcat/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step7_VideoNanRemove/'

    column_counter = numpy.genfromtxt(fname='Step6_NanAnalysis.csv', dtype=int, delimiter=',')[0:-1]
    threshold = column_counter[0] * 0.1

    file_list = FoldSearcher(load_path)
    last_column = None
    for file_path in tqdm.tqdm(file_list[4::5]):
        fold_path = file_path[0:-file_path[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path, save_path)):
            os.makedirs(fold_path.replace(load_path, save_path))

        file_data = numpy.genfromtxt(fname=file_path, dtype=float, delimiter=',')

        selected_data = None
        for index in range(numpy.shape(file_data)[1]):
            if column_counter[index] > threshold: continue
            if selected_data is None:
                selected_data = file_data[:, index:index + 1]
            else:
                selected_data = numpy.concatenate([selected_data, file_data[:, index:index + 1]], axis=-1)

        nan_flag = False
        with open(file_path.replace(load_path, save_path), 'w') as file:
            for indexX in range(numpy.shape(selected_data)[0]):
                for indexY in range(numpy.shape(selected_data)[1]):
                    if indexY != 0: file.write(',')
                    if selected_data[indexX][indexY] != selected_data[indexX][indexY]:
                        file.write(str(last_column[indexY]))
                        nan_flag = True
                    else:
                        file.write(str(selected_data[indexX][indexY]))
                if not nan_flag:
                    last_column = selected_data[indexX]
                file.write('\n')
        # exit()
