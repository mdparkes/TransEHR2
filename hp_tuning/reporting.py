"""Rendering tuning results as tables.

Reuses the table model in :mod:`jmir_reporting.tables`, so a tuning table prints to a terminal
and writes to Word through exactly the same path the manuscript tables use. These are working
tables rather than manuscript ones -- they exist so the sweep can be read at a glance and so
the choice of each hyperparameter is on record next to the numbers it was made from.
"""

import csv
import math
import os

from typing import Any, Dict, List, Optional

from jmir_reporting.tables import Table, build_document, render_text

from hp_tuning.results import compare_arms, rank_hyperparameter
from hp_tuning.spec import SELECTION_CRITERIA


# Supporting columns shown alongside each criterion's selection metric. The selection metric is
# always first; the rest are context, and a column whose key is absent from every row is
# dropped rather than filled with dashes.
SUPPORTING_METRICS = {
    'pretrain': [
        ('Optimization_Loss', 'Optimization loss'),
        ('Generator_Loss', 'Generator'),
        ('Discriminator_Loss', 'Discriminator'),
        ('THP_Loss', 'THP total'),
        ('THP_NLL_Loss', 'THP NLL'),
        ('THP_Type_Loss', 'THP type'),
        ('THP_Time_Loss', 'THP time'),
    ],
    'mortality': [
        ('AUPRC', 'AUPRC'),
        ('AUROC', 'AUROC'),
        ('F1_Score', 'F1'),
        ('Accuracy', 'Accuracy'),
        ('Loss_Cross_Entropy', 'Cross-entropy'),
    ],
}

STATUS_TEXT = {
    'pending': 'not run',
    'unreadable': 'unreadable',
    'nan': 'non-finite',
}


def format_metric(value: Optional[float]) -> str:
    """Render a metric for a table cell.

    Args:
        value: The metric, or None.

    Returns:
        Four decimal places for anything in a readable range, scientific notation for the very
        small values the THP time loss reaches, and an em dash for a missing number.
    """
    if value is None:
        return '—'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    if value != 0 and abs(value) < 1e-3:
        return f'{value:.3e}'
    return f'{value:.4f}'


def format_grid_value(value: Any) -> str:
    """Render a hyperparameter value for a row label.

    Args:
        value: The value.

    Returns:
        Its ``%g`` form for floats, so 2e-05 stays 2e-05 rather than becoming 0.00002.
    """
    if value is None:
        return 'null'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _used_columns(criterion: str, rankings: List[Dict[str, Any]]) -> List[tuple]:
    """Pick the metric columns that at least one row can fill.

    Args:
        criterion: The selection criterion.
        rankings: The rankings the table will show.

    Returns:
        The (key, heading) pairs to use, selection metric first.
    """
    candidates = SUPPORTING_METRICS[criterion]
    selection_metric = SELECTION_CRITERIA[criterion]['metric']
    present = set()
    for ranking in rankings:
        for result in ranking['results']:
            present.update(result.scores.keys())
    used = [(key, heading) for key, heading in candidates
            if key in present or key == selection_metric]
    return used or [(selection_metric, selection_metric.replace('_', ' '))]


def build_arm_table(
        manifest: Dict[str, Any],
        arm: str,
        criterion: str,
        number: int
) -> Optional[Table]:
    """Build the table of one arm's results under one selection criterion.

    Args:
        manifest: A loaded manifest.
        arm: The encoding arm.
        criterion: 'pretrain' or 'mortality'.
        number: The table number used in the caption.

    Returns:
        The table, or None if no hyperparameter on this arm uses the criterion.
    """
    hyperparameters = [
        name for name, entry in manifest['grid'].items()
        if entry['select_on'] == criterion
    ]
    if not hyperparameters:
        return None

    rankings = [rank_hyperparameter(manifest, arm, name) for name in hyperparameters]
    columns = _used_columns(criterion, rankings)
    spec = SELECTION_CRITERIA[criterion]

    if criterion == 'pretrain':
        caption = (
            f'Pretraining validation losses by hyperparameter value, {arm} encoding arm, '
            f'{manifest["fold"]}.'
        )
    else:
        caption = (
            f'Mortality validation performance by hyperparameter value, {arm} encoding arm, '
            f'{manifest["fold"]}.'
        )

    table = Table(
        number=number,
        caption=caption,
        stub_head='Hyperparameter and value',
        columns=[heading for _, heading in columns] + ['Trial'],
    )

    direction_word = 'lowest' if spec['direction'] == 'min' else 'highest'
    selected_note = table.add_footnote(
        f'Selected value: the {direction_word} {spec["metric"].replace("_", " ")}, marked *.'
    )
    table.add_footnote(
        'Every other tuned hyperparameter is held at its default, so each value is measured '
        'against one shared all-defaults centre rather than against a full factorial.'
    )

    for ranking in rankings:
        heading = f"{ranking['hyperparameter']}<sup>{selected_note}</sup>"
        if not ranking['complete']:
            heading += ' (incomplete)'
        table.add_category(heading)

        for result in ranking['results']:
            label = format_grid_value(result.grid_value)
            if ranking['best'] is not None and result is ranking['best']:
                label = f'*{label}*'
            if result.is_usable:
                cells = [format_metric(result.scores.get(key)) for key, _ in columns]
                cells.append(result.name)
            else:
                cells = [STATUS_TEXT.get(result.status, result.status)]
                cells += [''] * (len(columns) - 1)
                cells.append(result.name)
            table.add_row(label, cells, level=1)

    return table


def build_arm_comparison_table(
        manifest: Dict[str, Any],
        number: int,
        criterion: str = 'mortality'
) -> Optional[Table]:
    """Build the head-to-head table of the encoding arms at their shared centres.

    Args:
        manifest: A loaded manifest.
        number: The table number used in the caption.
        criterion: The criterion to compare on.

    Returns:
        The table, or None if there is only one arm.
    """
    if len(manifest['arms']) < 2:
        return None

    comparison = compare_arms(manifest, criterion)
    columns = _used_columns(criterion, [comparison])
    spec = SELECTION_CRITERIA[criterion]

    table = Table(
        number=number,
        caption=(
            f'Encoding arm comparison at the shared all-defaults centre, {manifest["fold"]}, '
            f'selected on {spec["metric"].replace("_", " ")}.'
        ),
        stub_head='Encoding arm',
        columns=[heading for _, heading in columns] + ['Trial'],
    )
    table.add_footnote(
        'The centres differ only in POSITION_ENCODING: same data, same hyperparameters, same '
        'frozen frequency ladder. The rotary arm makes the attention score a function of the '
        'time gap; the additive arm adds the encoding to the embedding, where it shares '
        'channels with content and stays entangled with absolute time.'
    )
    table.add_footnote(
        'Compared at the centre rather than at each arm\'s assembled optimum. This is the gap '
        'the revision plan names: a confirmation stage running each arm once at its own tuned '
        'settings would close it for four runs.'
    )

    for result in comparison['results']:
        label = str(result.grid_value)
        if comparison['best'] is not None and result is comparison['best']:
            label = f'*{label}*'
        if result.is_usable:
            cells = [format_metric(result.scores.get(key)) for key, _ in columns]
        else:
            cells = [STATUS_TEXT.get(result.status, result.status)] + [''] * (len(columns) - 1)
        cells.append(result.name)
        table.add_row(label, cells)

    return table


def build_all_tables(manifest: Dict[str, Any], first_number: int = 1) -> List[Table]:
    """Build every table for a sweep: one per arm per criterion, plus the arm comparison.

    Args:
        manifest: A loaded manifest.
        first_number: Number to give the first table.

    Returns:
        The tables, in the order they should appear.
    """
    tables = []
    number = first_number
    for arm in manifest['arms']:
        for criterion in ('pretrain', 'mortality'):
            table = build_arm_table(manifest, arm, criterion, number)
            if table is not None:
                tables.append(table)
                number += 1
    comparison = build_arm_comparison_table(manifest, number)
    if comparison is not None:
        tables.append(comparison)
    return tables


def write_csv(manifest: Dict[str, Any], path: str) -> str:
    """Write every trial result as one long CSV row per (arm, hyperparameter, value).

    The Word tables are for reading; this is for anything that wants to plot or re-check the
    sweep without parsing a document.

    Args:
        manifest: A loaded manifest.
        path: Destination CSV path.

    Returns:
        The path written.
    """
    metric_keys = []
    for keys in SUPPORTING_METRICS.values():
        metric_keys += [key for key, _ in keys]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['arm', 'hyperparameter', 'value', 'is_default', 'criterion', 'selection_metric',
             'selection_value', 'selected', 'status', 'trial', 'detail'] + metric_keys
        )
        for arm in manifest['arms']:
            for hyperparameter, entry in manifest['grid'].items():
                ranking = rank_hyperparameter(manifest, arm, hyperparameter)
                for result in ranking['results']:
                    writer.writerow([
                        arm,
                        hyperparameter,
                        format_grid_value(result.grid_value),
                        result.grid_value == entry['default'],
                        ranking['criterion'],
                        ranking['metric'],
                        '' if result.value is None else result.value,
                        ranking['best'] is not None and result is ranking['best'],
                        result.status,
                        result.name,
                        result.detail,
                    ] + [result.scores.get(key, '') for key in metric_keys])
    return path


def write_tables(
        manifest: Dict[str, Any],
        docx_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        print_text: bool = True
) -> List[str]:
    """Build and emit every table for a sweep.

    Args:
        manifest: A loaded manifest.
        docx_path: Where to write the Word document, or None to skip it.
        csv_path: Where to write the CSV, or None to skip it.
        print_text: Whether to render the tables to stdout as aligned text.

    Returns:
        The paths written.
    """
    tables = build_all_tables(manifest)
    written = []

    if print_text:
        for table in tables:
            print()
            render_text(table)
            print()

    if docx_path and tables:
        os.makedirs(os.path.dirname(os.path.abspath(docx_path)), exist_ok=True)
        build_document(tables, docx_path)
        written.append(docx_path)

    if csv_path:
        written.append(write_csv(manifest, csv_path))

    return written
