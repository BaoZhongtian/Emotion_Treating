import os
import tqdm
import librosa
import numpy
from scipy import signal
from Tools import FoldSearcher

if __name__ == '__main__':
    load_path = 'D:/PythonProject/IEMOCAP_Data/Step2_SelectIntersection/Audio/'
    save_path = 'D:/PythonProject/IEMOCAP_Data/Step3_SpectrumGeneration/'
    file_list = FoldSearcher(load_path)

    m_bands = 40
    s_rate = 16000
    win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
    hop_length = int(0.010 * s_rate)  # Window shift  10ms
    n_fft = win_length

    for file_path in tqdm.tqdm(file_list):
        y, sr = librosa.load(path=file_path, sr=s_rate)
        D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                   window=signal.hamming, center=False)) ** 2
        S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
        gram = librosa.power_to_db(S, ref=numpy.max)
        gram = numpy.transpose(gram, (1, 0))

        fold_path = file_path[0:-file_path[::-1].find('\\')]
        if not os.path.exists(fold_path.replace(load_path, save_path)):
            os.makedirs(fold_path.replace(load_path, save_path))

        with open(file_path.replace(load_path, save_path).replace('wav', 'csv'), 'w') as file:
            for indexX in range(numpy.shape(gram)[0]):
                for indexY in range(numpy.shape(gram)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(gram[indexX][indexY]))
                file.write('\n')
        # exit()
