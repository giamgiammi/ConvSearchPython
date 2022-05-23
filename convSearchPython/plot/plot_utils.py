"""Utils for plot package"""
import argparse
from pathlib import Path
from typing import Iterable, Tuple

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def argparse_positive_int(value):
    val = int(value)
    if val <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not >= 0")
    return val


def argparse_positive_int_tuple_or_none(value: str):
    if value is None:
        return None
    return tuple(argparse_positive_int(p.strip(' \t()')) for p in value.split(','))


def argparse_path_or_none(value):
    return Path(value) if value is not None else None


def plot_bars(x, y, data, save_path, size, tight, title, subtitle):
    fig = plt.figure(figsize=size)
    if subtitle != '':
        plt.suptitle(title)
        plt.title(subtitle)
    else:
        plt.title(title)
    sns.barplot(x=x, y=y, data=data)
    if tight:
        plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_bars_sub(x, y, rows, cols, datas: Iterable[Tuple[DataFrame, str]], size, tight, path, suptitle):
    fig, axs = plt.subplots(rows, cols, figsize=size)

    if suptitle is not None:
        fig.suptitle(suptitle)

    i = 0
    j = 0
    for d, name in datas:
        axs[i, j].set_title('conv {}'.format(name))
        sns.barplot(x=x, y=y, data=d, ax=axs[i, j])

        j = j + 1
        if j >= cols:
            j = 0
            i = i + 1

    if tight:
        fig.set_tight_layout(True)
    fig.savefig(path)
    plt.close(fig)