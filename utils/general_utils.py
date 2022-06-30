import re
from PyPDF2 import PdfFileMerger
from IPython import get_ipython


def natural_sort(l, reverse=False):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


def merge_pdfs(pdf_paths: list, output: str):
    """
    Merge multiple pdfs into one
    :param pdfs: list of PDF Path's to merge
    :param output: Output file name
    :return: None
    """
    merger = PdfFileMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    merger.write(output)
    merger.close()


def in_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
