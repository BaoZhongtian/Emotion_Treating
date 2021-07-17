import tqdm
import numpy
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step5_VideoConcat/'

    file_list = FoldSearcher(load_path)
    total_data, column_counter = [], None
    for file_path in tqdm.tqdm(file_list):
        file_data = numpy.genfromtxt(fname=file_path, dtype=float, delimiter=',')
        total_data.extend(file_data)
        if column_counter is None:
            column_counter = numpy.zeros(numpy.shape(total_data)[1])

        for indexX in range(numpy.shape(file_data)[0]):
            for indexY in range(numpy.shape(file_data)[1]):
                if file_data[indexX][indexY] != file_data[indexX][indexY]:
                    column_counter[indexY] += 1

    with open('Step6_NanAnalysis.csv', 'w') as file:
        for sample in column_counter:
            file.write(str(int(sample)) + ',')

    # total_data = numpy.array(total_data)
    # neo_total_data = []
    # for indexY in range(numpy.shape(column_counter)[0]):
    #     if column_counter[indexY] > 0.1 * len(total_data): continue
    #     if len(neo_total_data) == 0:
    #         neo_total_data = total_data[:, indexY:indexY + 1]
    #     else:
    #         neo_total_data = numpy.concatenate([neo_total_data, total_data[:, indexY:indexY + 1]], axis=1)
