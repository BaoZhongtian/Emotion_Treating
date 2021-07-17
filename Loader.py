import os
import json
import torch
import tqdm
import numpy
from torch.utils.data import Dataset, DataLoader
from GLOBAL_PARAMETER import IEMOCAP_JSON_PATH, EMOTION_LABEL, AVAILABLE_DEVICE


class IEMOCAP_Dataset_SingleMedia(Dataset):
    def __init__(self, treat_data):
        self.data, self.label = [], []
        for treat_sample in treat_data:
            if 'Audio' in treat_sample.keys(): self.data.append(treat_sample['Audio'])
            if 'Video' in treat_sample.keys(): self.data.append(treat_sample['Video'])
            self.label.append(EMOTION_LABEL[treat_sample['Emotion']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class Collate_IEMOCAP_SingleMedia:
    def __call__(self, batch):
        batch_data = [_[0] for _ in batch]
        batch_label = [_[1] for _ in batch]
        assert len(batch_data) == len(batch_label)
        tensor_label = torch.LongTensor(batch_label).to(AVAILABLE_DEVICE)

        tensor_data, tensor_seq = [], torch.LongTensor([len(_) for _ in batch_data]).to(AVAILABLE_DEVICE)
        batch_max_length = max(tensor_seq)
        for treat_sample in batch_data:
            tensor_data.append(
                numpy.concatenate([treat_sample, numpy.zeros([batch_max_length - numpy.shape(treat_sample)[0],
                                                              numpy.shape(treat_sample)[1]])]))

        tensor_data = torch.FloatTensor(tensor_data).to(AVAILABLE_DEVICE)
        return tensor_data, tensor_seq, tensor_label


class Loader_IEMOCAP:
    def __init__(self, audio_flag=False, video_flag=False, consider_part=None, fast_part_load_flag=False):
        if consider_part is None:
            consider_part = ['improve', 'script']

        self.audio_total_data, self.video_total_data = None, None
        if audio_flag:
            self.audio_total_data = []

            for session_index in range(1, 6):
                for gender_name in ['Female', 'Male']:
                    for part_name in consider_part:
                        current_data = json.load(
                            open(os.path.join(IEMOCAP_JSON_PATH, 'IEMOCAP_Audio_%s_%s_Session%d.json' % (
                                part_name, gender_name, session_index)), 'r'))
                        self.audio_total_data.extend(current_data)
                        print('Load Audio Session', session_index, gender_name, part_name, 'Completed',
                              len(self.audio_total_data))
                if fast_part_load_flag and session_index > 1: break

            print('Audio Data Load Completed')
            print('Total', len(self.audio_total_data), 'Samples.')

        if video_flag:
            self.video_total_data = []

            for session_index in range(1, 6):
                for gender_name in ['Female', 'Male']:
                    for part_name in consider_part:
                        current_data = json.load(
                            open(os.path.join(IEMOCAP_JSON_PATH, 'IEMOCAP_Video_%s_%s_Session%d.json' % (
                                part_name, gender_name, session_index)), 'r'))
                        self.video_total_data.extend(current_data)
                        print('Load Video Session', session_index, gender_name, part_name, 'Completed',
                              len(self.video_total_data))
                if fast_part_load_flag and session_index > 1: break

        if self.audio_total_data is None and self.video_total_data is None:
            assert RuntimeError('Please Input at least one Media.')

    def TrainValTestSeparate(self, appoint_session, appoint_gender, batch_size):
        assert 1 <= appoint_session <= 5

        if self.audio_total_data is not None:
            audio_train_data, audio_val_data, audio_test_data = [], [], []
            for treat_sample in self.audio_total_data:
                if int(treat_sample['Session'][-1]) != appoint_session:
                    audio_train_data.append(treat_sample)
                if int(treat_sample['Session'][-1]) == appoint_session and treat_sample['Gender'] != appoint_gender:
                    audio_val_data.append(treat_sample)
                if int(treat_sample['Session'][-1]) == appoint_session and treat_sample['Gender'] == appoint_gender:
                    audio_test_data.append(treat_sample)

            print(len(audio_train_data), len(audio_val_data), len(audio_test_data))
            if self.video_total_data is None:
                train_loader = self.CollateBatch_SingleMedia(audio_train_data, batch_size, shuffle_flag=True)
                val_loader = self.CollateBatch_SingleMedia(audio_val_data, batch_size, shuffle_flag=False)
                test_loader = self.CollateBatch_SingleMedia(audio_test_data, batch_size, shuffle_flag=False)
                return train_loader, val_loader, test_loader

        if self.video_total_data is not None:
            video_train_data, video_val_data, video_test_data = [], [], []
            for treat_sample in self.video_total_data:
                if int(treat_sample['Session'][-1]) != appoint_session:
                    video_train_data.append(treat_sample)
                if int(treat_sample['Session'][-1]) == appoint_session and treat_sample['Gender'] != appoint_gender:
                    video_val_data.append(treat_sample)
                if int(treat_sample['Session'][-1]) == appoint_session and treat_sample['Gender'] == appoint_gender:
                    video_test_data.append(treat_sample)
            print(len(video_train_data), len(video_val_data), len(video_test_data))

            if self.audio_total_data is None:
                train_loader = self.CollateBatch_SingleMedia(video_train_data, batch_size, shuffle_flag=True)
                val_loader = self.CollateBatch_SingleMedia(video_val_data, batch_size, shuffle_flag=False)
                test_loader = self.CollateBatch_SingleMedia(video_test_data, batch_size, shuffle_flag=False)

                return train_loader, val_loader, test_loader

    def CollateBatch_SingleMedia(self, treat_data, batch_size, shuffle_flag):
        treat_dataset = IEMOCAP_Dataset_SingleMedia(treat_data)
        treat_loader = DataLoader(dataset=treat_dataset, batch_size=batch_size, shuffle=shuffle_flag,
                                  collate_fn=Collate_IEMOCAP_SingleMedia())
        return treat_loader


if __name__ == '__main__':
    dataset = Loader_IEMOCAP(video_flag=True, consider_part=['improve'], fast_part_load_flag=True)
    dataset.TrainValTestSeparate(appoint_session=1, appoint_gender='Female', batch_size=8)
