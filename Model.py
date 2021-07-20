import torch
import numpy
from GLOBAL_PARAMETER import CLASS_NUMBER, AVAILABLE_DEVICE


class Baseline_BLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Baseline_BLSTM, self).__init__()
        self.blstm_layer = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, bidirectional=True, bias=True, batch_first=True)
        self.predict_layer = torch.nn.Linear(in_features=hidden_size * 2, out_features=CLASS_NUMBER)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, batch_data, batch_seq, batch_label=None):
        assert batch_data.size()[0] == batch_seq.size()[0]

        blstm_output, blstm_state = self.blstm_layer(batch_data)
        final_state = []
        # print(numpy.shape(blstm_output))
        for index in range(batch_seq.size()[0]):
            final_state.append(blstm_output[index][batch_seq[index] - 1:batch_seq[index]])
        final_state = torch.cat(final_state, dim=0)
        predict = self.predict_layer(final_state)

        if batch_label is not None:
            loss = self.loss_function(input=predict, target=batch_label)
            return loss
        else:
            return predict


class Baseline_CNN(torch.nn.Module):
    def __init__(self, attention_hidden_size):
        super(Baseline_CNN, self).__init__()
        self.attention_hidden_size = attention_hidden_size
        self.__BuildNetwork()

    def __BuildNetwork(self):
        self.conv_1st = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU()
        )
        self.conv_2nd = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU()
        )
        self.conv_3rd = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )
        self.attention_layer = torch.nn.Linear(in_features=self.attention_hidden_size, out_features=1)
        self.predict_layer = torch.nn.Linear(in_features=self.attention_hidden_size, out_features=CLASS_NUMBER)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def ApplyVanillaAttention(self, batch_data, batch_seq=None):
        transposed_result = torch.permute(batch_data, [0, 2, 1, 3])
        reshaped_result = transposed_result.reshape([transposed_result.size()[0], transposed_result.size()[1],
                                                     transposed_result.size()[2] * transposed_result.size()[3]])
        attention_weight = self.attention_layer(reshaped_result).squeeze()

        if batch_seq is not None:
            assert batch_data.size()[0] == batch_seq.size()[0]
            batch_seq = batch_seq.cpu().numpy()
            batch_seq = [int(_) for _ in numpy.ceil(batch_seq / 8)]
            attention_mask = []
            for index in range(len(batch_seq)):
                attention_mask.append(numpy.concatenate([numpy.zeros(batch_seq[index]),
                                                         numpy.ones(numpy.max(batch_seq) - batch_seq[index])]))
            attention_mask = torch.BoolTensor(attention_mask).to(AVAILABLE_DEVICE)
            attention_weight = torch.masked_fill(attention_weight, mask=attention_mask, value=-9999)

        attention_weight_softmax = attention_weight.softmax(dim=-1)
        attention_weight_softmax_repeat = attention_weight_softmax.unsqueeze(-1).repeat(
            1, 1, self.attention_hidden_size)
        # print(numpy.shape(attention_weight_softmax_repeat), numpy.shape(reshaped_result))
        weighted_result = torch.multiply(attention_weight_softmax_repeat, reshaped_result).sum(dim=1)
        return weighted_result, attention_weight_softmax

    def forward(self, batch_data, batch_seq, batch_label=None):
        batch_data = batch_data.unsqueeze(1)

        conv1st_result = self.conv_1st(batch_data)
        conv2nd_result = self.conv_2nd(conv1st_result)
        conv3rd_result = self.conv_3rd(conv2nd_result)

        attention_result, attention_weight = self.ApplyVanillaAttention(conv3rd_result, batch_seq)
        predict = self.predict_layer(attention_result)

        if batch_label is not None:
            loss = self.loss_function(input=predict, target=batch_label)
            return loss
        else:
            return predict
