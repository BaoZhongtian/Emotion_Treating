import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    load_path = 'D:/PythonProject/CurrentData/MultiDialogue_Result_Again_Result/'
    accuracy_line1 = []
    for filename in os.listdir(load_path):
        data = numpy.genfromtxt(fname=os.path.join(load_path, filename), dtype=int, delimiter=',')
        counter = 0
        for index in range(len(data)):
            if data[index][0] == data[index][1]: counter += 1
        accuracy_line1.append(counter / len(data))

    load_path = 'D:/PythonProject/CurrentData/MultiDialogue_Result_OnlyQuestion_Result/'
    accuracy_line2 = []
    for filename in os.listdir(load_path):
        data = numpy.genfromtxt(fname=os.path.join(load_path, filename), dtype=int, delimiter=',')
        counter = 0
        for index in range(len(data)):
            if data[index][0] == data[index][1]: counter += 1
        accuracy_line2.append(counter / len(data))

    accuracy_line2 = accuracy_line2[0:len(accuracy_line1)]

    plt.plot(accuracy_line1, label='Question&Answer')
    plt.plot(accuracy_line2, label='Only Question')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Per 1K train batch number')
    plt.show()
