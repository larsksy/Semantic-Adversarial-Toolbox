import os

def get_class(idx, dataset):

    if dataset == 'imagenet':
        path = os.path.join(os.getcwd(), './classes/imagenet.txt')
    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    with open(path) as f:
        idx2label = eval(f.read())

    return idx2label[idx]
