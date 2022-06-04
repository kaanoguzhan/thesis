import re
from PyPDF2 import PdfFileMerger


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def merge_pdfs(pdfs: list, output: str):
    """
    Merge multiple pdfs into one
    :param pdfs: list of pdf Path's to merge
    :param output: output file name
    :return: None
    """
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(output)
    merger.close()
