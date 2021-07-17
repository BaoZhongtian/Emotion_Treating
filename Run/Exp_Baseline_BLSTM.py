import os
from Model import Baseline_BLSTM
from Loader import Loader_IEMOCAP
from Template import Template_Basic_Train_Method
from GLOBAL_PARAMETER import IEMOCAP_SAVE_PATH

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = Loader_IEMOCAP(audio_flag=True)

    for appoint_session in range(1, 6):
        for appoint_gender in ['Female', 'Male']:
            save_path = IEMOCAP_SAVE_PATH + '/Baseline_BLSTM/Session%d_%s/' % (
                appoint_session, appoint_gender)
            train_loader, val_loader, test_loader = dataset.TrainValTestSeparate(
                appoint_session=1, appoint_gender='Female', batch_size=8)
            model = Baseline_BLSTM(input_size=40, hidden_size=128)
            Template_Basic_Train_Method(model, train_loader, val_loader, test_loader, save_path)
