from sqlalchemy.future import create_engine
from sqlalchemy import text
import pandas as pd
import datetime
import time
import csv
from pathlib import Path
from shutil import copyfile


MAIN_PATT = re.compile(r"(MAINCIT)+", re.UNICODE)
CIT_PATT = re.compile(r"(CIT)+", re.UNICODE)


def load_dataset(path):
    sep = "\t" if ".tsv" in path else "␞"
    names = (
        None
        if ".tsv" in path
        else [
            "uuid",
            "lang",
            "ctd_mid",
            "ctg_mid",
            "ctd_aid",
            "ctg_aid",
            "published",
            "context",
        ]
    )
    df = pd.read_csv(path, sep=sep, names=names, error_bad_lines=False)
    return df


def save_dataset(df, path):
    df.to_csv(path, sep="\t", index=False)


def make_sample(*, xling_path, adj_path):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    assert Path(xling_path).is_file()
    assert Path(adj_path).is_file()

    xling_names = [
        "uuid",
        "ctd_mag_id",
        "adj_mid_str",
        "ctg_mid",
        "ctd_aid",
        "adj_aid_str",
        "ctg_aid",
        "context",
    ]
    adj_names = [
        "lang",
        "uuid",
        "citing_mid",
        "cited_mid",
        "citing_aid",
        "cited_aid",
        "bibitem_str",
    ]

    sample = []
    debt = 0
    leftovers = []

    for name, group in adj_contexts_df.groupby(["citing_aid"]):
        assert (
            group.dropna(subset=["lang"]).shape[0] > 0
        )  # make sure at least one item has a language

        rows = {}
        lang_ids = []
        non_lang_ids = []

        for i, row in enumerate(group.iterrows()):

            lang, _2, _3, _4, citing_aid, _6, _7 = row[1]
            rows[str(i)] = row[1].tolist()

            if pd.isna(lang):
                non_lang_ids.append(i)
            else:
                lang_ids.append(i)

        num_samples = 0

        for j, lang_id in enumerate(lang_ids):
            if non_lang_ids:
                idx, idx_val, _ = get_nearest_id(lang_id, non_lang_ids)
            else:
                print(
                    f"Document with citing id: {citing_aid} has a debt of {str(len(lang_ids)-num_samples)} samples."
                )
                debt += len(lang_ids) - num_samples
                break

            row = rows[str(idx_val)]
            if pd.isna(row[0]):
                sample.append(row)
                del non_lang_ids[idx]
                num_samples += 1
            else:
                pass

            if j == len(lang_ids) - 1:
                if non_lang_ids:
                    leftovers += [rows[str(_id)] for _id in non_lang_ids]

    print("Total debt:", debt)
    print("Length of samples: ", len(sample))
    print("Length of contexts:", contexts_df.shape[0])

    output_path = adj_path.replace(".", f"_sample_{timestamp}.")
    output_df = pd.DataFrame(sample, columns=adj_names)
    output_df.to_csv(output_path, index=False, sep="\t")
    print(f"Saved to path: {output_path}")

    return output_path


def filter_db(*, db_path, citations_path):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    assert Path(db_path).is_file()

    # create database engine
    db_uri = "sqlite:///{}".format(db_path)
    db_engine = create_engine(db_uri)

    # search statement
    stmt = text(
        "select uuid, citing_mag_id, cited_mag_id, citing_arxiv_id, cited_arxiv_id, bibitem_string from bibitem;"
    )

    # TODO
    if "sample" in citations_path:
        names = [
            "uuid",
            "lang",
            "aid",
            "published",
            "category",
            "journal_ref",
            "bibitem_str",
        ]
    else:
        names = [
            "lang",
            "uuid",
            "citing_mid",
            "cited_mid",
            "citing_aid",
            "cited_aid",
            "bibitem_str",
        ]
    citations_df = pd.read_csv(
        citations_path, sep="\t", names=names, low_memory=False, quoting=csv.QUOTE_NONE
    )
    citations_df = citations_df.set_index("uuid")

    # filter database entries
    with db_engine.connect() as conn:
        rows = conn.execute(stmt)

        i = 0
        dropped = 0
        kept = 0

        change = False

        for row in rows:
            (
                uuid,
                citing_mad_id,
                cited_mag_id,
                citing_arxiv_id,
                cited_arxiv_id,
                bibitem,
            ) = row

            if uuid not in citations_df.index:
                del_stmt = text("DELETE FROM bibitem WHERE uuid=:uuid")
                conn.execute(del_stmt, {"uuid": uuid})
                dropped += 1
                change = True
            else:
                kept += 1

            i += 1
            if i % 10000 == 0:
                print(f"At i={str(i)}, dropped {str(dropped)} and kept {str(kept)}.")

                if change:
                    conn.commit()
                    change = False

    print(f'Finished filtering "{db_path}"')


def normalize_citations(text):
    text = re.sub(MAIN_PATT, "", text)
    text = re.sub(CIT_PATT, "", text)
    return text


def remove_duplicates(path):
    assert Path(path).is_file()

    df = load_dataset(path)

    w = re.search("=\d_", path).group(0)[1]

    print("Staring shape:", df.shape)
    # normalize context column and save to new column
    df["context_norm"] = df["context"].apply(normalize_citations)
    # remove duplicates from new column
    unique_df = df.drop_duplicates(subset=["context_norm"])
    # remove new column
    unique_df = unique_df.drop(columns=["context_norm"])
    print("Ending shape:", unique_df.shape)
    output_path = path.replace(".", "_unique.")
    unique_df.to_csv(output_path, sep="\t", index=False)

    print(f"Saved file to path: {output_path}")

    return output_path


def make_mixed_column(contexts1, contexts2):
    return [1 if c in contexts1 else 0 for c in contexts2]


def label_mixed(*, xling_path, mono_path):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    assert Path(xling_path).is_files()
    assert Path(mono_path).is_files()

    xling_df = load_dataset(xling_path)
    mono_df = load_dataset(mono_path)

    xling_contexts_norm = xling_df["context"].apply(normalize_citations).tolist()
    mono_contexts_norm = mono_df["context"].apply(normalize_citations).tolist()

    xling_df["mixed"] = make_mixed_column(mono_contexts_norm, xling_contexts_mono)
    mono_df["mixed"] = make_mixed_column(xling_contexts_mono, mono_contexts_norm)

    output_xling_path = xling_path.replace(".", "_mixed.")
    output_mono_path = mono_path.replace(".", "_mixed.")
    save_dataset(xling_df, output_xling_path)
    save_dataset(mono_df, output_mono_path)

    print(f"Saved files to paths: {output_xling_path} and {output_mono_path}")

    return output_xling_path, output_mono_path


def format_file(*, path, xling_path=None):
    assert Path(path).is_file()
    if xling_path:
        assert Path(xling_path.is_file())

    if "sample" in path:
        names = ["uuid", "ctd_mid", "ctg_mid", "ctd_aid", "ctg_aid", "context"]
    else:
        names = [
            "uuid",
            "lang",
            "ctd_mid",
            "ctg_mid",
            "ctd_aid",
            "ctg_aid",
            "published",
            "category",
            "context",
        ]

    df = pd.read_csv(path, sep="␞", names=names, error_bad_lines=False)

    if xling_path:
        xling_df = pd.read_csv(xling_path, sep="\t").set_index("uuid")

    uuids_not_in_index = []
    rows = []

    for i, row in df.iterrows():
        uuid = row["uuid"]
        context = row["context"]
        ctd_mid = row["ctd_mag_id"]
        ctg_mid = row["ctg_mid"]
        ctd_aid = row["ctd_aid"]
        ctg_aid = row["ctg_aid"]
        context = row["context"]

        if xling_path:
            if uuid not in xling.index:
                uuids_not_in_index.append(uuid)
                continue

            in_lang_row = in_lang_df.loc[uuid]
            lang = in_lang_row["lang"]
            published = in_lang_row["published"]
            category = in_lang_row["category"]

            new_row = [
                uuid,
                lang,
                ctd_mid,
                ctg_mid,
                ctd_aid,
                ctg_aid,
                published,
                category,
                context,
            ]
            rows.append(new_row)
        else:
            new_row = [uuid, ctd_mid, ctg_mid, ctd_aid, ctg_aid, context]
            rows.append(new_row)

    output_df = pd.DataFrame(rows, columns=names)

    output_path = path.replace(".csv", ".tsv")
    save_dataset(output_df, output_path)

    print(f"Saved file to path: {output_path}")

    return output_path
