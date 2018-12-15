import torch
import copy
import math
from torch import optim
from torch.autograd import Variable

from tqdm import tqdm

def meta_train(model, params, use_gpu):
    best_model_weights = None
    best_avg_acc = 0
    best_avg_loss = math.inf
    patience_counter = 0

    for epoch in range(params.n_epochs):
        progress_bar = tqdm(params.train_loader, desc="Epoch {}".format(epoch))
        progress_bar.set_description("Epoch {}".format(epoch))
        model.train()

        if params.scheduler:
            params.scheduler.step()

        avg_train_loss = math.inf
        avg_train_acc = 0
        avg_val_acc = 0
        for (idx, train_batch), (val_idx, val_batch) in zip(enumerate(progress_bar), enumerate(params.valid_loader)):

            inputs, targets = train_batch
            val_inputs, val_targets = val_batch

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                val_inputs = val_inputs.cuda()
                val_targets = val_targets.cuda()

            params.optimizer.zero_grad()

            # Training of the model
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = model(inputs)

            loss, train_accuracy = params.criterion(predictions, targets)
            loss.backward()
            params.optimizer.step()

            avg_train_acc += train_accuracy.cpu().data.numpy() / params.n_episodes

            # Validation of the training
            val_inputs, val_targets = Variable(val_inputs), Variable(val_targets)
            val_predictions = model(val_inputs)

            _, val_acc = params.criterion(val_predictions, val_targets)

            avg_train_loss += loss.cpu().data.numpy() / params.n_episodes
            avg_val_acc += val_acc.cpu().data.numpy() / params.n_episodes

            progress_bar.set_postfix({'loss': avg_train_loss, 't_acc':avg_train_acc, 'v_acc':avg_val_acc})

        # Deep copy the best model
        if avg_train_loss < best_avg_loss:
            patience_counter = 0
            best_avg_loss = avg_train_loss
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= params.PATIENCE_LIMIT:
            return best_model_weights

    return best_model_weights
