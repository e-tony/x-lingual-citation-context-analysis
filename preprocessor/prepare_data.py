"""
Usage: python prepare_data.py -db ../data/samplerefs.db -cp -ms -xp ../data/contexts_with_uuids_window=0_full.csv -sp ../data/in_lang_docs_all_bibitems.tsv
"""

from argparse import ArgumentParser
from prepare_data_helper import *


def main(*, xling_path, citations_path, db_path, make_sample=False, sample_path=None):
    if make_sample and not sample_path:
        print("Making sample...")
        mono_path = make_sample(xling_path=xling_path, adj_path=sample_path)
    elif not sample_path:
        raise Error('Must supply a "sample_path".')

    print("Filtering db...")
    filter_db(db_path=db_path, citations_path=citations_path)

    print("Removing duplicates...")
    xling_path = remove_duplicates(xling_path)
    mono_path = remove_duplicates(mono_path)

    print("Labeling mixed contexts...")
    xling_path, mono_path = label_mixed(xling_path=xling_path, mono_path=mono_path)

    print("Formatting files...")
    xling_path = format_file(path=xling_path, xling_path=xling_path)
    mono_path = format_file(path=mono_path, xling_path=xling_path)

    print("Finished preparing data.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-db",
        "--db_path",
        dest="db_path",
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--citations_path",
        dest="citations_path",
        default=None,
    )
    parser.add_argument(
        "-ms",
        "--make_sample",
        dest="make_sample",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-xp",
        "--xling_path",
        dest="xling_path",
        default=None,
    )
    parser.add_argument(
        "-sp",
        "--sample_path",
        dest="sample_path",
        default=None,
    )
    args = parser.parse_args()

    main(
        xling_path=args.xling_path,
        citations_path=args.citations_path,
        db_path=args.db_path,
        make_sample=args.make_sample,
        sample_path=args.sample_path,
    )
