from torch import nn


def _parse_network(network, outputs, pretrained, **kwargs):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or nn.Module
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    pretrained : bool
        Use pretrained parameters as in model_zoo

    Returns
    -------
    results: list of nn.Module (the same size as len(outputs))

    """
    l, n = len(outputs), len(outputs[0])
    results = [[] for _ in range(l)]
    if isinstance(network, str):
        from model.model_zoo import get_model
        network = get_model(network, pretrained=pretrained, **kwargs).features

    # helper func
    def recursive(pos, block, arr, j):
        if j == n:
            results[pos].append([block])
            return
        child = list(block.children())
        results[pos].append(child[:arr[j]])
        if pos + 1 < l: results[pos + 1].append(child[arr[j] + 1:])
        recursive(pos, child[arr[j]], arr, j + 1)

    block = list(network.children())

    for i in range(l):
        pos = outputs[i][0]
        if i == 0:
            results[i].append(block[:pos])
        elif i < l:
            results[i].append(block[outputs[i - 1][0] + 1: pos])
        recursive(i, block[pos], outputs[i], 1)

    for i in range(l):
        results[i] = nn.Sequential(*[item for sub in results[i] for item in sub if sub])
    return results
