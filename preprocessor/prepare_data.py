from argparse import ArgumentParser
from prepare_data_helper import *


def main(*, xling_path, sample_path, citations_path, db_path, make_sample):
    # run extract_contexts_mod_orig.py

    if make_sample:
        print('Making sample...')
        db_path = make_sample(xling_path=xling_path, adj_path=sample_path)
    else:
        db_path = db_path

    print('Filtering db...')
    output_path = filter_db(db_path=db_path, citations_path=citations_path)

    print('Removing duplicates...')
    output_path = remove_duplicates(output_path)

    print('Labeling mixed contexts...')
    xling_path, mono_path = label_mixed(xling_path=, mono_path=)

    print('Formatting files...')
    xling_path = format_file(path=, xling_path=)
    mono_path = format_file(path=, xling_path=)

    print'Finished prepareing data.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-db',
        '--db_path',
        dest='db_path',
        default=None,
        help='')
    parser.add_argument(
        '-cp',
        '--citations_path',
        dest='citations_path',
        default=None,
        help='')
    parser.add_argument(
        '-ms',
        '--make_sample',
        dest='make_sample',
        default=False,
        action='store_true',
        help='')
    parser.add_argument(
        '-xp',
        '--xling_path',
        dest='xling_path',
        default=None,
        help='')
    parser.add_argument(
        '-sp',
        '--sample_path',
        dest='sample_path',
        default=None,
        help='')
    args = parser.parse_args()

    main(
        xling_path=args.xling_path, 
        sample_path=args.sample_path, 
        citations_path=args.citations_path, 
        db_path=args.db_path, 
        make_sample=args.make_sample
        )

    # python prepare_data.py -db ../data/unarXive/samplerefs.db -xp ../data/unarXive/code/contexts_with_uuids_window=0_full.csv -sp ../data/citation_contexts/in_lang_docs_all_bibitems.tsv
