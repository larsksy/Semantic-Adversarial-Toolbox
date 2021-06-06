from tqdm import tqdm


def get_tqdm(desc, total, purge=False):
    """Wrapper for TQDM. Prevents bugs related to multiple instances of TQDM.

    :param desc: Description for tqdm.
    :param total: Total number of iterations for tqdm.
    :param purge: If True, closes all previous tqdm objects to avoid problems of multiple instances.
    :return: TQDM object
    """
    # Close old instances to prevent bugs
    if purge:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
    return tqdm(desc=desc, ascii=True, total=total)


def get_tqdm_iterable(iterable, desc, total, purge=False):
    """Wrapper for iterable TQDM. Prevents bugs related to multiple instances of TQDM.

    :param iterable: The iterable object to make a tqdm object from.
    :param desc: Description for tqdm.
    :param total: Total number of iterations for tqdm.
    :param purge: If True, closes all previous tqdm objects to avoid problems of multiple instances.
    :return: TQDM object
    """
    # Close old instances to prevent bugs
    if purge:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
    return tqdm(enumerate(iterable, 0), ascii=True, desc=desc, total=total)
