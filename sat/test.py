import torch
from collections import OrderedDict


def test(model, testloader, criterion, device='cpu', t=None, best_acc=None):
    """Test accuracy and loss of model on a dataloader object.

    :param model: The model to test.
    :param testloader: Dataloader object with images to test.
    :param criterion: The loss function for measuring model loss.
    :param device: 'cuda' if running on gpu, 'cpu' otherwise
    :param t: Optional tqdm object for showing progress in terminal.
    :param best_acc: Optional parameter to keep track of best accuracy in case code is run in multiple iterations.

    :return Accuracy of test
    """

    # Initialization of variables
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forwards pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print results to terminal
            if t is not None and testloader.num_workers == 0:
                od = OrderedDict()
                od['type'] = 'test'
                od['loss'] = test_loss / (batch_idx + 1)
                od['acc'] = '{:.3f}'.format(100. * correct / total)
                if best_acc is not None:
                    od['test_acc'] = best_acc
                od['iter'] = '%d/%d' % (total, len(testloader.dataset))
                t.set_postfix(ordered_dict=od)
                t.update(inputs.shape[0])
            else:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, total, len(testloader.dataset)))

    acc = 100.*correct/total
    return acc
