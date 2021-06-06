from collections import OrderedDict


def train(model, dataloader, criterion, optimizer, device='cpu', t=None, best_acc=None):
    """One iteration of model training. Intentionally kept generic to increase versatility.

    :param model: The model to train.
    :param dataloader: Dataloader object of training dataset.
    :param criterion: The loss function for measuring model loss.
    :param device: 'cuda' if running on gpu, 'cpu' otherwise
    :param t: Optional tqdm object for showing progress in terminal.
    :param best_acc: Optional parameter to keep track of best accuracy in case code is run in multiple iterations.

    :return: Training accuracy
    """

    # Initialize variables
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forwards pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Backpropagation
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Keep track of loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print results to terminal
        if t is not None and dataloader.num_workers == 0:
            od = OrderedDict()
            od['type'] = 'train'
            od['loss'] = train_loss / (batch_idx + 1)
            od['acc'] = 100. * correct / total
            od['test_acc'] = best_acc if best_acc is not None else None
            od['iter'] = '%d/%d' % (correct, total)
            t.set_postfix(ordered_dict=od)
            t.update(inputs.shape[0])
        else:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100. * correct / total
