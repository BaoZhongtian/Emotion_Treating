import os
from Model import Baseline_CNN
from Loader import Loader_IEMOCAP
from Template import Template_Basic_Train_Method
from GLOBAL_PARAMETER import IEMOCAP_SAVE_PATH, FEATURE_SHAPE, AVAILABLE_DEVICE

Appoint_Media = 'Audio'

if __name__ == '__main__':
    dataset = Loader_IEMOCAP(audio_flag=Appoint_Media == 'Audio', video_flag=Appoint_Media == 'Video')

    for appoint_session in range(1, 6):
        for appoint_gender in ['Female', 'Male']:
            save_path = IEMOCAP_SAVE_PATH + '/Baseline_BLSTM_%s/Session%d_%s/' % (
                Appoint_Media, appoint_session, appoint_gender)
            train_loader, val_loader, test_loader = dataset.TrainValTestSeparate(
                appoint_session=appoint_session, appoint_gender=appoint_gender, batch_size=8)
            model = Baseline_CNN(attention_hidden_size=320)
            Template_Basic_Train_Method(model, train_loader, val_loader, test_loader, save_path)
