"""Probes for the Table 1 launcher and its report job.

Both are shell scripts and untracked, like every other launcher here, so a clone has neither
and these skip. What they check is scope: the launcher exists to re-run Table 1 without
touching any other experiment or table, and every way it could reach further -- writing every
config, finetuning every task, running the full report -- still submits and completes.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_redo_configs import CHARLSON_EXPERIMENTS


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, 'run_table1.sh')
REPORT_JOB = os.path.join(REPO, 'SLURM', 'slurm_report_table1.sh')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(LAUNCHER) and os.path.exists(REPORT_JOB)),
    reason='run_table1.sh and SLURM/ are untracked; nothing to check in this checkout'
)


def read(path):
    with open(path) as handle:
        return handle.read()


def commands(text):
    """The script with its comment lines removed."""
    return '\n'.join(line for line in text.splitlines()
                      if not line.lstrip().startswith('#'))


@pytest.fixture(scope='module')
def launcher():
    return commands(read(LAUNCHER))


@pytest.fixture(scope='module')
def report_job():
    return commands(read(REPORT_JOB))


def test_only_the_charlson_arms_are_trained(launcher):
    match = re.search(r'^EXPERIMENTS=\(([^)]*)\)', launcher, re.M)
    assert match
    assert tuple(int(n) for n in match.group(1).split()) == CHARLSON_EXPERIMENTS


def test_only_their_configs_are_written(launcher):
    """Without --only the generator rewrites every redo config."""
    calls = launcher.count('scripts/generate_redo_configs.py')
    assert calls == launcher.count('--only "${EXPERIMENTS[@]}"') == 2


def test_only_mortality_is_finetuned(launcher):
    assert 'TASKS="mortality"' in launcher
    assert 'TASKS="${TASKS}"' in launcher


def test_the_episode_list_is_restricted_and_checked_before_submission(launcher):
    assert '--cohort "${CHARLSON_COHORT}"' in launcher
    assert 'CHARLSON_COHORT="diagnosis_history"' in launcher
    assert '--expect_n "${CHARLSON_EXPECT_N}"' in launcher
    assert launcher.index('compute_charlson_index.py') < launcher.index('--job-name="train')


def test_a_dump_waits_for_its_training_to_succeed(launcher):
    assert 'afterok:"${TRAIN_ID}"' in launcher


def test_the_report_job_builds_table_1_and_nothing_else(report_job):
    scripts = set(re.findall(r'scripts/(\w+\.py)', report_job))
    assert scripts == {'run_charlson_logistic_regression.py', 'report_results_tables.py'}
    assert re.findall(r'--cohorts (\w+)', report_job) == ['charlson']


def test_the_regression_reads_the_list_the_training_reads(launcher, report_job):
    assert '--cohort_episodes "${CHARLSON_EPISODES}"' in report_job
    assert 'CHARLSON_EPISODES="${CHARLSON_EPISODES}"' in launcher


def test_every_reporter_is_pointed_at_the_model_tree(report_job):
    for block in report_job.split('stage '):
        if 'report_results_tables.py' in block:
            assert '--model-dir "${MODEL_DIR}"' in block
        if 'run_charlson_logistic_regression.py' in block:
            assert '--model_dir "${MODEL_DIR}"' in block
