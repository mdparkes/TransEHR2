#!/usr/bin/env python3
"""Report the diagnosis label counts and prevalences, one column per cohort.

Counts the positive phenotype labels of each episode's ICU stay over the extracted arrays and
renders them as a supplementary table, alphabetized by diagnosis. Each cell is the count and
the prevalence within that column's cohort:

    count (prevalence)

Labels are read from `phenotype.npy` rather than from the phenotyping listfiles, so the table
describes the episodes that survived extraction -- the same population the models are trained
and evaluated on. The listfiles supply only the column order, which is the order the label
matrix was written in.

The two heart failure categories are reported separately. ICD-9 codes reach their phenotype
through HCUP CCS 2015 and ICD-10 codes through HCUP CCSR 2024, and the two vocabularies name
that category differently; the label matrix carries both names as distinct columns and the
models predict them as distinct labels, so pooling them here would describe a different task.

Usage:
    python report_diagnosis_prevalence.py --data_dir data/
    python report_diagnosis_prevalence.py --table-number S3 \\
        --output tables/tableS3_diagnosis_labels.docx --csv tables/tableS3.csv

One fold's train, val and test partitions cover the cohort once, so the default reads fold0.
Passing more folds double counts, and the run stops if it detects that.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from TransEHR2.data.cohorts import has_any_historical_text, has_value_history
from TransEHR2.data.preprocessing import load_dataset, load_episode_ids
from reporting.jmir.tables import Table, build_document, render_text


TASK_PREFIX = 'phenotyping'

DEFAULT_CAPTION = 'Diagnosis labels'


def episode_cohorts(dataset) -> dict:
    """Boolean membership arrays for every reported cohort.

    The cohorts are the two the experiments run on: any pre-admission record, and any
    pre-admission text record.

    Args:
        dataset: A loaded `MixedDataset`.

    Returns:
        Dict of cohort key to (n_episodes,) boolean array.
    """
    hist = dataset.max_history_len_steps
    n = np.asarray(dataset.val_masks).shape[0]
    return {
        'all': np.ones(n, dtype=bool),
        'history': has_value_history(dataset.val_masks, hist),
        'text': has_any_historical_text(dataset.val_masks,
                                        dataset.val_text_indicators, hist),
    }


# (key, column heading). The heading completes with the cohort size when the table is built.
COHORT_COLUMNS = (
    ('all', 'All patients'),
    ('history', 'Patients with a pre-admission record'),
    ('text', 'Patients with pre-admission text'),
)


def phenotype_names(data_dir: str, fold: str) -> list:
    """The label column order, read from a fold's phenotyping listfile.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold name.

    Returns:
        Ordered list of label names.

    Raises:
        FileNotFoundError: If no phenotyping listfile is present in the fold.
    """
    fold_dir = os.path.join(data_dir, fold)
    for partition in ('train', 'val', 'test'):
        for name in (f'{TASK_PREFIX}_{partition}_listfile.unfiltered.csv',
                     f'{TASK_PREFIX}_{partition}_listfile.csv'):
            path = os.path.join(fold_dir, name)
            if os.path.exists(path):
                columns = list(pd.read_csv(path, index_col=0, nrows=0).columns)
                return [column for column in columns if column != 'period_length']
    raise FileNotFoundError(
        f'no phenotyping listfile under {fold_dir}, so the label columns cannot be named.'
    )


def collect(data_dir: str, folds, splits, extracted_history_len_steps=None) -> dict:
    """Label matrix and cohort membership over every requested partition.

    Raises:
        SystemExit: If no partition was read, or if a patient appears in more than one, which
            means the partitions overlap and those episodes would be counted twice.
    """
    labels = []
    cohorts = {key: [] for key, _ in COHORT_COLUMNS}
    patients = []

    for fold in folds:
        for split in splits:
            path = os.path.join(data_dir, fold, split)
            if not os.path.isdir(path):
                print(f'  {fold}/{split}: not found, skipping', file=sys.stderr)
                continue
            dataset = load_dataset(path,
                                   extracted_history_len_steps=extracted_history_len_steps)
            phenotype = np.asarray(dataset.phenotype)
            labels.append(phenotype)
            for key, mask in episode_cohorts(dataset).items():
                cohorts[key].append(mask)
            # Ids are patient_id * 1000 + episode_number; see MixedDataReader.
            patients.append(load_episode_ids(path, len(phenotype)) // 1000)
            print(f'  {fold}/{split}: {len(phenotype)} episodes')

    if not labels:
        raise SystemExit('No partitions were read. Check --data_dir, --folds and --splits.')

    merged = {
        'labels': np.concatenate(labels, axis=0),
        'patients': np.concatenate(patients),
        'cohorts': {key: np.concatenate(parts) for key, parts in cohorts.items()},
    }

    _, counts = np.unique(merged['patients'], return_counts=True)
    if (counts > 1).any():
        raise SystemExit(
            f'{int((counts > 1).sum())} patients appear in more than one partition, so those '
            f'episodes would be counted twice. One fold covers the cohort once -- pass a '
            f'single fold.'
        )
    return merged


def build_table(data, names, number: str, caption: str) -> Table:
    """Render the counts and prevalences as a manuscript table.

    Args:
        data: The dict `collect` returns.
        names: Ordered label names, one per column of the label matrix.
        number: Table number for the caption.
        caption: Caption text, without the "Table N." prefix.

    Returns:
        The assembled `Table`.

    Raises:
        SystemExit: If the label matrix and the listfile disagree on the number of labels, or
            if a cohort is empty.
    """
    labels = data['labels']
    if labels.shape[1] != len(names):
        raise SystemExit(
            f'the extracted label matrix has {labels.shape[1]} columns but the phenotyping '
            f'listfile names {len(names)}. They describe different extractions.'
        )

    sizes = {key: int(data['cohorts'][key].sum()) for key, _ in COHORT_COLUMNS}
    for key, heading in COHORT_COLUMNS:
        if sizes[key] == 0:
            raise SystemExit(f'the "{heading}" cohort is empty, so its prevalences are '
                             f'undefined.')

    columns = [f'{heading} (n={sizes[key]:,})' for key, heading in COHORT_COLUMNS]
    table = Table(number, caption, 'Label', columns,
                  short_columns=[heading for _, heading in COHORT_COLUMNS])
    table.add_footnote('Each cell gives the number of episodes carrying the label, with the '
                       'prevalence within that column\'s cohort in parentheses.')

    for index in np.argsort(names, kind='stable'):
        cells = []
        for key, _ in COHORT_COLUMNS:
            positive = int(labels[data['cohorts'][key], index].sum())
            cells.append(f'{positive:,} ({positive / sizes[key]:.2f})')
        table.add_row(names[index], cells)
    return table


def write_csv(path: str, data, names) -> None:
    """Write the same numbers as a CSV, one row per label and cohort."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('label,cohort,episodes,cohort_size,prevalence\n')
        for index in np.argsort(names, kind='stable'):
            for key, _ in COHORT_COLUMNS:
                mask = data['cohorts'][key]
                size = int(mask.sum())
                positive = int(data['labels'][mask, index].sum())
                handle.write(f'"{names[index]}",{key},{positive},{size},'
                             f'{positive / size:.6f}\n')
    print(f'Wrote {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Tabulate diagnosis label counts and prevalences by cohort'
    )
    parser.add_argument('--data_dir', default='data',
                        help='Directory holding the fold subdirectories (default: data)')
    parser.add_argument('--folds', nargs='+', default=['fold0'],
                        help='Folds to read. One fold covers the cohort once; passing more '
                             'double counts, and the run stops if it detects that.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Partitions within each fold (default: train val test)')
    parser.add_argument('--table-number', default='S3', help='Table number for the caption')
    parser.add_argument('--caption', default=DEFAULT_CAPTION, help='Caption text')
    parser.add_argument('--output', default=None,
                        help='Word document to write (default: print only)')
    parser.add_argument('--csv', default=None, help='Also write the numbers to this CSV')
    parser.add_argument('--caption-style', default=None,
                        help='Paragraph style for the caption in the Word document')
    parser.add_argument('--extracted-history-len-steps', type=int, default=None,
                        help='Width of the history region in the extracted arrays. Only needed '
                             'for datasets written before the layout was recorded in metadata.')
    args = parser.parse_args(argv)

    print(f'Reading {args.data_dir}: folds {" ".join(args.folds)}, '
          f'splits {" ".join(args.splits)}')
    data = collect(args.data_dir, args.folds, args.splits,
                   args.extracted_history_len_steps)
    names = phenotype_names(args.data_dir, args.folds[0])

    table = build_table(data, names, args.table_number, args.caption)
    render_text(table)

    if args.csv:
        write_csv(args.csv, data, names)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        build_document([table], args.output, caption_style=args.caption_style)
        print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
