"""Probes for the redo coordinator and its report job.

These are shell scripts, so nothing here runs them. What is checkable without a scheduler is
the part that goes wrong quietly: three files name the same set of experiments and the same set
of tables, and a set that drifts produces a run that submits, completes and is missing a column
or a table nobody notices until the manuscript is assembled.

The dependency structure is checked as text for the same reason. A dump chained with `afterany`
instead of `afterok` would dump a half-trained model, and the CSV it writes is not
distinguishable from a complete one downstream.
"""

import os
import re
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_redo_configs import CHARLSON_EXPERIMENT, EXPERIMENTS, experiment_name


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDINATOR = os.path.join(REPO, 'run_redo.sh')
REPORT_JOB = os.path.join(REPO, 'SLURM', 'slurm_report_redo.sh')

# Shell scripts are not tracked in this repository -- `/*.sh` and `SLURM/` are both ignored, so
# what runs a job carries the absolute paths of the machine it runs on. A clone therefore has
# neither of these files and the probes below have nothing to read; skipping is correct rather
# than failing, since their absence is the repository's convention and not a fault.
pytestmark = pytest.mark.skipif(
    not (os.path.exists(COORDINATOR) and os.path.exists(REPORT_JOB)),
    reason='run_redo.sh and SLURM/ are untracked; nothing to check in this checkout'
)


def read(path):
    with open(path) as handle:
        return handle.read()


def commands(text):
    """The script with its comment lines removed, for probes about what it runs.

    A script that documents what it does not do would otherwise satisfy a probe looking for the
    absence of a command.
    """
    return '\n'.join(line for line in text.splitlines()
                      if not line.lstrip().startswith('#'))


@pytest.fixture(scope='module')
def coordinator():
    return read(COORDINATOR)


@pytest.fixture(scope='module')
def report_job():
    return read(REPORT_JOB)


# ------------------------------------------------------------------------------------------
# The experiment set
# ------------------------------------------------------------------------------------------

def test_the_coordinator_trains_every_generated_experiment(coordinator):
    """An experiment written but never submitted has no column and no error."""
    match = re.search(r'^EXPERIMENTS=\(([^)]*)\)', coordinator, re.M)
    assert match, 'run_redo.sh declares no EXPERIMENTS'
    submitted = [int(n) for n in match.group(1).split()]
    assert submitted == [entry[0] for entry in EXPERIMENTS]


def test_the_regression_arm_is_not_submitted_as_a_training_job(coordinator):
    """Experiment 29 has no config and no model; a training task for it would fail every
    fold, and its column comes from the regression the report job runs."""
    match = re.search(r'^EXPERIMENTS=\(([^)]*)\)', coordinator, re.M)
    assert 29 not in [int(n) for n in match.group(1).split()]
    assert 'run_charlson_logistic_regression.py' in commands(read(REPORT_JOB))


def test_the_charlson_list_is_written_before_the_training_is_submitted(coordinator):
    """Experiment 28 reads COHORT_EPISODES at load time, so a list written by the report job
    would be too late -- the training would already have run on every episode."""
    index_at = coordinator.index('compute_charlson_index.py')
    submit_at = coordinator.index('--array="0-$((N_FOLDS - 1))')
    assert index_at < submit_at
    assert str(CHARLSON_EXPERIMENT) in coordinator


# ------------------------------------------------------------------------------------------
# The job chain
# ------------------------------------------------------------------------------------------

def test_a_dump_waits_for_its_training_to_succeed(coordinator):
    """afterany here would dump a half-trained model into a CSV that reads like any other."""
    assert 'afterok:"${TRAIN_ID}"' in coordinator
    assert '--kill-on-invalid-dep=yes' in coordinator


def test_the_report_runs_even_when_a_dump_fails(coordinator):
    """A partly finished set is when the report is most useful, and the tables mark a missing
    column rather than hiding it."""
    assert 'afterany:"${DEPENDENCY}"' in coordinator


def test_every_stage_of_the_report_runs_even_after_one_fails(report_job):
    """The stages are independent, so a missing dump should cost one table rather than all."""
    assert 'FAILED+=' in report_job
    assert report_job.rstrip().endswith('[ "${#FAILED[@]}" -eq 0 ]'), (
        'the report job must still exit non-zero when a stage failed'
    )


# ------------------------------------------------------------------------------------------
# The tables
# ------------------------------------------------------------------------------------------

# Every cohort declared by the results writer has to be reported, or a whole table is silently
# absent: the writer defaults to all three, and naming them one at a time is what makes the
# separate captions and table numbers possible.
@pytest.mark.parametrize('cohort', ['charlson', 'textsubset', 'historysubset'])
def test_each_results_cohort_is_reported(report_job, cohort):
    assert f'--cohorts {cohort}' in commands(report_job)


def test_the_reported_cohorts_are_the_ones_the_writer_declares(report_job):
    from report_results_tables import COHORT_KEYS
    reported = set(re.findall(r'--cohorts (\w+)', commands(report_job)))
    assert reported == set(COHORT_KEYS)


@pytest.mark.parametrize('script', [
    'report_results_tables.py',
    'report_carryforward_comparison.py',
    'report_diagnosis_prevalence.py',
    'report_historic_diagnosis_audit.py',
    'plot_history_distributions.py',
])
def test_every_writer_the_redo_needs_is_invoked(report_job, script):
    assert script in commands(report_job)
    assert os.path.exists(os.path.join(REPO, script)), f'{script} is invoked but absent'


def test_the_figure_is_built_on_the_cohort_the_tables_report(report_job):
    """plot_history_distributions defaults to any_history. The figure accompanies the tables
    computed on the text cohort, and the default would produce a figure about a different
    population with nothing in the output saying so."""
    body = commands(report_job)
    line = next(l for l in body.splitlines() if 'plot_history_distributions.py' in l)
    assert '--cohort any_text' in line, f'no cohort named: {line.strip()}'


def test_the_audit_is_restricted_to_the_evaluated_cohort(report_job):
    """audit_historic_diagnoses defaults to every extracted episode. Supplementary Table 7 and
    Table 5 both rest on it, and both report rates for the population the models were evaluated
    on -- unrestricted, the audit answers a question about a population no result covers."""
    body = commands(report_job)
    block = next(b for b in body.split('stage ') if 'audit_historic_diagnoses.py' in b)
    assert '--cohort any_text' in block, f'the audit names no cohort: {block.strip()[:120]}'


def test_the_audit_and_the_text_experiments_share_a_cohort(report_job):
    """The audit's population has to be the one the arms it describes were trained on, or the
    named-versus-unnamed rates it reports are computed over different stays than the model
    scores they are compared with."""
    from generate_redo_configs import EXPERIMENTS
    text_arms = {entry[2] for entry in EXPERIMENTS if entry[0] in (20, 24)}
    assert text_arms == {'any_text'}
    body = commands(report_job)
    block = next(b for b in body.split('stage ') if 'audit_historic_diagnoses.py' in b)
    assert f'--cohort {text_arms.pop()}' in block


def test_the_inputs_are_built_before_the_tables_that_read_them(report_job):
    """Each of these writes what the table after it reads, and running them the other way
    round produces a table from the previous run's inputs rather than an error."""
    for earlier, later in (('audit_historic_diagnoses.py', 'report_historic_diagnosis_audit.py'),
                           ('audit_historic_diagnoses.py', 'stratify_predictions_by_history.py'),
                           ('stratify_predictions_by_history.py',
                            'report_carryforward_comparison.py')):
        body = commands(report_job)
        assert body.index(earlier) < body.index(later), (
            f'{later} runs before {earlier}, which builds its input'
        )


def test_the_carryforward_arms_are_resolved_rather_than_spelled_out(report_job):
    """Both arms are named by number through experiment_name, so a rename of an experiment
    cannot leave this pointed at a directory that no longer exists."""
    body = commands(report_job)
    assert 'experiment_name(20)' in body
    assert 'experiment_name(24)' in body
    for number in (20, 24):
        assert experiment_name(number) not in body, (
            f'experiment {number} is spelled out as well as resolved'
        )


def test_the_tuning_tables_are_not_regenerated_here(report_job):
    """Supplementary Tables 4-6 come from the Phase 2 runs and are already written; rebuilding
    them from this job would need the tuning trees, which it does not read."""
    assert 'report_tuning_tables.py' not in commands(report_job)


# ------------------------------------------------------------------------------------------
# The download
# ------------------------------------------------------------------------------------------

def test_the_download_lands_in_the_repository_it_is_run_from(coordinator):
    """The absolute path on SDRE is not the absolute path on the laptop, so only the source
    side may be absolute; an absolute destination would write the cluster's tree locally."""
    block = coordinator[coordinator.index('print_rsync() {'):coordinator.index('EOF\n}')]
    # Continuations first: a wrapped rsync ends its line with a backslash, not its destination.
    joined = re.sub(r'\\+\s*\n\s*', ' ', block)
    invocations = [line for line in joined.splitlines() if 'rsync -avz' in line]
    assert invocations, 'the download block runs no rsync'
    for line in invocations:
        destination = line.split()[-1]
        assert destination.startswith('./'), (
            f'destination {destination!r} is not relative to the local repository'
        )


# Every reporter defaults to a relative ./models, which is not the tree the training wrote to:
# the experiment configs carry an absolute MODEL_DIR. A reporter pointed at the wrong tree finds
# no experiment directories and reports empty columns rather than failing.
@pytest.mark.parametrize('script,flag', [
    ('report_results_tables.py', '--model-dir'),
    ('stratify_predictions_by_history.py', '--model-dir'),
    ('run_charlson_logistic_regression.py', '--model_dir'),
])
def test_every_reporter_is_pointed_at_the_tree_the_runs_wrote_to(report_job, script, flag):
    body = commands(report_job)
    for block in body.split('stage '):
        if script in block:
            assert f'{flag} "${{MODEL_DIR}}"' in block, (
                f'{script} is invoked without {flag}, so it reads its own default'
            )


def test_the_report_job_stops_when_the_model_tree_is_absent(report_job):
    """Reported as an error rather than as nine empty tables."""
    assert '! -d "${MODEL_DIR}"' in report_job


def test_the_charlson_paths_are_named_rather_than_defaulted(coordinator):
    """compute_charlson_index writes to its own default when --write_cohort is bare, which
    agrees with CHARLSON_DIR only by coincidence; experiment 28 reads the path this names."""
    assert '--write_cohort "${CHARLSON_EPISODES}"' in coordinator
    assert '--output "${CHARLSON_CSV}"' in coordinator


def test_the_download_takes_the_predictions_from_the_model_tree(coordinator):
    """dump_finetuned_predictions writes under {model_dir}/{experiment}/{fold}/{task}/, not to
    a directory of its own, and that tree also holds every checkpoint -- so the download has to
    name the tree and filter it rather than pulling a predictions directory that does not
    exist."""
    block = coordinator[coordinator.index('print_rsync() {'):coordinator.index('EOF\n}')]
    assert '${MODEL_DIR}/' in block
    assert "--include='*_finetuned_output.csv'" in block
    assert "--exclude='*'" in block, 'without a trailing exclude the filter pulls the weights'
    assert 'misc/predictions' not in block


def test_the_download_names_the_account_as_well_as_the_host(coordinator):
    """A bare hostname resolves to whatever user the laptop's ssh config defaults to, which is
    not necessarily the account the files belong to."""
    match = re.search(r'SDRE_HOST="\$\{SDRE_HOST:-([^}]+)\}"', coordinator)
    assert match, 'run_redo.sh names no SDRE host'
    assert '@' in match.group(1), f'{match.group(1)!r} names a host but no account'

    block = coordinator[coordinator.index('print_rsync() {'):coordinator.index('EOF\n}')]
    assert 'sdre:' not in block, 'the host is spelled out rather than taken from SDRE_HOST'


def test_the_download_covers_what_the_reporting_reads(coordinator):
    """The point of the second block is rerunning the reporting on the laptop, which needs the
    inputs and not just the finished tables."""
    block = coordinator[coordinator.index('print_rsync() {'):coordinator.index('EOF\n}')]
    for directory in ('tables', 'misc/stratified_carryforward',
                      'misc/historic_diagnoses', 'charlson'):
        assert directory in block, f'{directory} is not in the download command'


def test_the_arrays_that_cannot_be_downloaded_are_named(coordinator):
    """Two reports read the extraction rather than a dump, so the download cannot make them
    reproducible locally and says so instead of appearing complete."""
    block = coordinator[coordinator.index('print_rsync() {'):coordinator.index('EOF\n}')]
    assert 'Supplementary Table 3' in block and 'Supplementary Figure 1' in block


# ------------------------------------------------------------------------------------------
# Consistency with the configs
# ------------------------------------------------------------------------------------------

def test_the_coordinator_names_the_scripts_it_submits(coordinator):
    for path in re.findall(r'^(?:TRAIN|DUMP|REPORT)_SCRIPT="([^"]+)"', coordinator, re.M):
        assert os.path.exists(os.path.join(REPO, path)), f'{path} is submitted but absent'


def test_the_history_window_defaults_to_the_extraction_capacity(coordinator):
    """The sweep reports a curve; the experiments run at the window that was chosen from it.
    A default that drifted from the dataset config would crop silently."""
    match = re.search(r'HISTORY_LEN_STEPS="\$\{HISTORY_LEN_STEPS:-(\d+)\}"', coordinator)
    assert match, 'run_redo.sh sets no default history window'
    dataset = yaml.safe_load(open(os.path.join(
        REPO, 'TransEHR2', 'configs', 'datasets', 'mimic4.yaml')))
    assert int(match.group(1)) == dataset['MAX_HISTORY_LEN_STEPS']
