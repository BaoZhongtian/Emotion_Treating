import torch
import numpy
from GLOBAL_PARAMETER import CLASS_NUMBER


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
