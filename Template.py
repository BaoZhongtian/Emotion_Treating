import os
import torch
import numpy
from Tools import ProgressBar
from GLOBAL_PARAMETER import AVAILABLE_DEVICE, AVAILABLE_GPU_NUMBER, LEARNING_RATE, ADAM_EPSILON, MAX_GRAD_NORM


def Template_Basic_Train_Method(model, train_loader, val_loader, test_loader, save_path, episode_number=100):
    if os.path.exists(save_path): return
    os.makedirs(save_path)

    model.to(AVAILABLE_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    if AVAILABLE_GPU_NUMBER > 1: model = torch.nn.DataParallel(model)

    model.zero_grad()
    pbar = ProgressBar(n_total=len(train_loader) * episode_number, desc='Training')
    for episode_index in range(episode_number):
        episode_loss = 0.0

        model.train()
        loss_file = open(os.path.join(save_path, '%04d-Loss.csv' % episode_index), 'w')
        for batch_index, [batch_data, batch_seq, batch_label] in enumerate(train_loader):
            loss = model(batch_data, batch_seq, batch_label)
            if AVAILABLE_GPU_NUMBER > 1: loss = loss.mean()

            pbar(episode_index * len(train_loader) + batch_index, {'loss': loss.item()})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            model.zero_grad()

            episode_loss += loss.item()
            loss_file.write(str(loss.item()) + '\n')
        print('\nEpisode', episode_index, 'Total Loss =', episode_loss)
        loss_file.close()

        model.eval()
        for eval_part in ['Eval', 'Test']:
            predict_file = open(os.path.join(save_path, '%04d-%s.csv' % (episode_index, eval_part)), 'w')
            if eval_part == 'Eval':
                loader = val_loader
            else:
                loader = test_loader

            for batch_index, [batch_data, batch_seq, batch_label] in enumerate(loader):
                predict = model(batch_data, batch_seq)
                predict = predict.detach().cpu().numpy()
                batch_label = batch_label.detach().cpu().numpy()
                for indexX in range(numpy.shape(predict)[0]):
                    predict_file.write(str(batch_label[indexX]))
                    for indexY in range(numpy.shape(predict)[1]):
                        predict_file.write(',' + str(predict[indexX][indexY]))
                    predict_file.write('\n')
            predict_file.close()
