#!/usr/bin/env python3
"""End-to-end smoke test of the Phase 2 pipeline, on the real data, on a GPU.

Everything in the Phase 2 infrastructure has been exercised against synthetic data or against
the shape of the code. This is the first thing that touches MIMIC-IV on a compute node, and
nothing expensive should be queued until it passes.

What it checks, in the order it checks it:

1. **Environment.** A GPU is visible, bf16 is supported, and the Accelerator resolves to a
   single non-distributed process. A stale ``ACCELERATE_*`` variable would otherwise turn one
   trial into eight copies fighting over one checkpoint directory, and the failure would look
   like corrupt checkpoints rather than like a misconfiguration.

2. **Data and the frequency ladder.** Loads the tuning fold and measures the largest temporal
   span the value and event encoders actually see, then checks the configured ladder bounds
   against it. This is open item 2 in the revision plan: ``VALUE_LADDER_P_MAX: 7.9e+6`` follows
   from a Δ_max of 125,073 h measured on the *pre*-re-extraction data, and the whole argument
   for re-spanning collapses if the new arrays disagree.

3. **Memory.** Runs two batches of pretraining at the real batch size for each encoding arm and
   reports peak VRAM. Single-GPU packing is a premise of the entire phase, not a measurement,
   until this passes.

4. **Pipeline.** Generates a miniature sweep -- both arms, one pretrain-selected and one
   mortality-selected hyperparameter -- and runs it all the way through: config generation,
   pretrain jobs, finetune jobs, reporting, table writing, and selection. Truncated to a few
   hundred episodes and two epochs, so it tests the plumbing rather than the model.

5. **Timing.** Extrapolates from the truncated run to a full-size trial and prints the
   ``--time`` value the real jobs should request. Requesting more than a job takes
   deprioritizes the whole group's queue position, so the sweep should be launched with a
   measured limit rather than the placeholder in the batch scripts.

Usage:
    python test_phase2_pipeline.py --work_dir /tmp/phase2_test
    python test_phase2_pipeline.py --work_dir /tmp/phase2_test --skip memory
    python test_phase2_pipeline.py --work_dir /tmp/phase2_test --only data

Nothing here writes to the real models/ tree: MODEL_DIR is rewritten to point inside
--work_dir, which the SLURM wrapper deletes afterwards.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from TransEHR2.utils import EPOCH_TIMING_PREFIX, STARTUP_TIMING_PREFIX


STAGES = ('environment', 'data', 'memory', 'pipeline', 'timing')

# The miniature sweep. One hyperparameter selected on pretraining loss and one selected on
# mortality, so both selection paths and both SLURM stages are exercised, at two values each so
# a ranking is a real comparison rather than a single point.
TEST_GRID = {
    'PRETRAIN_LEARNING_RATE': {'values': [0.002, 0.0002], 'select_on': 'pretrain'},
    'CMPNT_MASK_RATIO': {'values': [0.25, 0.5], 'select_on': 'mortality'},
}
TEST_ALIASES = {'PRETRAIN_LEARNING_RATE': 'lr', 'CMPNT_MASK_RATIO': 'cmask'}

# Scripts the pipeline stage invokes as subprocesses. Stage 1 imports each one so a missing
# dependency surfaces in the first seconds rather than in the subprocess that needs it.
PIPELINE_ENTRY_POINTS = (
    'generate_tuning_configs',
    'run_experiment',
    'report_tuning_results',
    'select_tuned_hyperparameters',
)


class Report:
    """Collects check outcomes so the summary can be read without scrolling the log."""

    def __init__(self):
        self.entries = []
        self.notes = []

    def record(self, stage, check, ok, detail=''):
        """Record one check and print it as it happens.

        Args:
            stage: The stage the check belongs to.
            check: What was checked.
            ok: True for pass, False for fail, None for a warning that is not a failure.
            detail: Supporting text.
        """
        self.entries.append((stage, check, ok, detail))
        marker = {True: 'PASS', False: 'FAIL', None: 'WARN'}[ok]
        print(f"  [{marker}] {check}" + (f"\n         {detail}" if detail else ''), flush=True)

    def note(self, text):
        """Record a line for the final summary that is not a pass or a fail."""
        self.notes.append(text)
        print(f"  ....   {text}", flush=True)

    @property
    def failed(self):
        """The checks that failed."""
        return [e for e in self.entries if e[2] is False]

    @property
    def warned(self):
        """The checks that warned."""
        return [e for e in self.entries if e[2] is None]

    def summarise(self):
        """Print the summary block and return the process exit status."""
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        passed = sum(1 for e in self.entries if e[2] is True)
        print(f"  {passed} passed, {len(self.warned)} warnings, {len(self.failed)} failed")
        for stage, check, ok, detail in self.entries:
            if ok is not True:
                marker = 'FAIL' if ok is False else 'WARN'
                print(f"\n  {marker}  [{stage}] {check}")
                if detail:
                    for line in detail.splitlines():
                        print(f"        {line}")
        if self.notes:
            print("\n  Measurements:")
            for text in self.notes:
                print(f"    - {text}")
        print("=" * 70)
        if self.failed:
            print("PHASE2_TEST_FAILED")
            return 1
        print("PHASE2_TEST_OK")
        return 0


def run_stage_environment(args, report):
    """Check that the compute node is what the single-GPU premise assumes.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
    """
    import torch

    print("\n" + "=" * 70)
    print("STAGE 1: environment")
    print("=" * 70)

    report.note(f"python {sys.version.split()[0]}, torch {torch.__version__}")

    # Import every script the pipeline stage shells out to, before anything trains. Each has a
    # __main__ guard, so importing only pulls its dependency chain -- which is the point: a
    # package missing from the environment is otherwise not discovered until that subprocess
    # runs, and the reporting scripts run last, after every pretrain and finetune.
    for module_name in PIPELINE_ENTRY_POINTS:
        try:
            __import__(module_name)
            report.record('environment', f'{module_name} imports', True)
        except Exception as error:
            report.record('environment', f'{module_name} imports', False,
                          f'{type(error).__name__}: {error}. Stage 4 invokes this script; the '
                          f'run would fail there instead of here. Check the environment '
                          f'against requirements.txt.')

    if not torch.cuda.is_available():
        report.record('environment', 'CUDA is available', False,
                      'No GPU visible. This test must run on a GPU node; submit it with '
                      'SLURM/slurm_test_phase2.sh rather than running it on a login node.')
        return
    report.record('environment', 'CUDA is available', True)

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    report.note(f"GPU: {props.name}, {total_gb:.0f} GB, {torch.cuda.device_count()} visible")

    report.record('environment', 'bf16 is supported', torch.cuda.is_bf16_supported(),
                  '' if torch.cuda.is_bf16_supported() else
                  'The runner asks for bf16 because that is what every previous experiment '
                  'ran under. Without it the runs would be fp32 and roughly double the '
                  'activation memory, which the single-GPU packing does not budget for.')

    stale = [name for name in
             ('ACCELERATE_USE_FSDP', 'ACCELERATE_MIXED_PRECISION', 'ACCELERATE_CONFIG_FILE',
              'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')
             if os.environ.get(name)]
    report.record('environment', 'no distributed environment variables set', not stale,
                  f"Set: {stale}. run_experiment.py will refuse to start. The SLURM scripts "
                  f"unset these; something else in the shell profile is putting them back."
                  if stale else '')

    from run_experiment import initialize_accelerator
    try:
        accelerator = initialize_accelerator('bf16')
        report.record('environment', 'Accelerator resolves to one non-distributed process',
                      True, f"device={accelerator.device}, "
                            f"mixed_precision={accelerator.mixed_precision}")
        report.record('environment', 'mixed precision is actually bf16',
                      accelerator.mixed_precision == 'bf16',
                      f"Got {accelerator.mixed_precision!r}. Nothing reads the accelerate "
                      f"config YAML under a bare python launch, so this must come from the "
                      f"constructor argument."
                      if accelerator.mixed_precision != 'bf16' else '')
    except Exception as error:
        report.record('environment', 'Accelerator resolves to one non-distributed process',
                      False, f"{type(error).__name__}: {error}")


def measure_spans(times, masks, chunk=4096):
    """Measure the largest temporal span within an episode, over valid timesteps only.

    Attention is all-pairs, so the gap the encoding has to represent is the span between the
    earliest and latest record in an episode, not the largest step-to-step increment. Both are
    returned because the step gap is what the P_MIN end of the ladder answers to.

    Args:
        times: (n_episodes, n_timesteps) array of timestamps in hours, memory-mapped.
        masks: (n_episodes, n_timesteps) array, 1.0 at a real record and 0.0 at padding.
        chunk: Episodes to bring into memory at a time.

    Returns:
        A dict with 'max_span', 'max_step_gap', 'min_step_gap', 'median_span' and
        'n_episodes_with_records'.
    """
    max_span = 0.0
    max_step = 0.0
    min_step = np.inf
    spans = []

    for start in range(0, times.shape[0], chunk):
        block_times = np.asarray(times[start:start + chunk], dtype=np.float64)
        block_valid = np.asarray(masks[start:start + chunk]) > 0.5
        if not block_valid.any():
            continue

        # Padding timestamps are stored as 0.0, which is a legitimate admission-relative time,
        # so they have to be excluded by the mask rather than by value.
        masked = np.where(block_valid, block_times, np.nan)
        with np.errstate(invalid='ignore'):
            lo = np.nanmin(masked, axis=1)
            hi = np.nanmax(masked, axis=1)
        episode_spans = hi - lo
        episode_spans = episode_spans[np.isfinite(episode_spans)]
        if episode_spans.size:
            spans.append(episode_spans)
            max_span = max(max_span, float(episode_spans.max()))

        for row_times, row_valid in zip(block_times, block_valid):
            observed = np.sort(row_times[row_valid])
            if observed.size < 2:
                continue
            steps = np.diff(observed)
            steps = steps[steps > 0]
            if steps.size:
                max_step = max(max_step, float(steps.max()))
                min_step = min(min_step, float(steps.min()))

    all_spans = np.concatenate(spans) if spans else np.zeros(0)
    return {
        'max_span': max_span,
        'max_step_gap': max_step,
        'min_step_gap': None if not np.isfinite(min_step) else min_step,
        'median_span': float(np.median(all_spans)) if all_spans.size else 0.0,
        'n_episodes_with_records': int(all_spans.size),
    }


def run_stage_data(args, report, base_config):
    """Load the tuning fold and check the ladder bounds against the data.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
        base_config: The loaded base experiment config.
    """
    import pickle

    print("\n" + "=" * 70)
    print("STAGE 2: data and the frequency ladder")
    print("=" * 70)

    with open(args.dataset_config, 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    fold_dir = os.path.join(dataset_config['DATA_DIR'], args.fold)

    if not os.path.isdir(fold_dir):
        report.record('data', f'{args.fold} exists', False, f'{fold_dir} is not a directory')
        return
    report.record('data', f'{args.fold} exists', True, fold_dir)

    train_dir = os.path.join(fold_dir, 'train')
    meta_path = os.path.join(train_dir, 'metadata.pkl')
    if not os.path.exists(meta_path):
        report.record('data', 'the fold has been extracted', False,
                      f'{meta_path} is missing. Run ./reextract.sh before this test.')
        return
    with open(meta_path, 'rb') as f_in:
        metadata = pickle.load(f_in)
    report.record('data', 'the fold has been extracted', True)

    text_embed_dim = metadata.get('text_embed_dim', 0)
    if base_config.get('USE_TEXT'):
        report.record('data', 'text embeddings are present', text_embed_dim > 0,
                      'text_embed_dim is 0. Run embed_text.py on a GPU node before tuning; '
                      'the base config has USE_TEXT: True and the run will refuse to start.'
                      if text_embed_dim == 0 else f'text_embed_dim = {text_embed_dim}')
        # The single-GPU memory budget assumes the embedding shrank from Llama's width to a
        # 768- or 1024-d encoder. A 4096 here means the swap did not happen.
        if text_embed_dim >= 2048:
            report.record('data', 'the text encoder was swapped', None,
                          f'text_embed_dim is {text_embed_dim}, which is a large-LLM width. '
                          f'Phase 1 replaced the encoder with bge-m3 at 1024. The embeddings '
                          f'may predate that change.')

    # --------------------------------------------------- the ladder, against the real gaps
    for stream, times_name, masks_name, p_min_key, p_max_key in (
            ('value', 'val_times', 'val_masks', 'VALUE_LADDER_P_MIN', 'VALUE_LADDER_P_MAX'),
            ('event', 'event_times', 'event_masks', 'EVENT_LADDER_P_MIN', 'EVENT_LADDER_P_MAX'),
    ):
        times_path = os.path.join(train_dir, f'{times_name}.npy')
        masks_path = os.path.join(train_dir, f'{masks_name}.npy')
        if not (os.path.exists(times_path) and os.path.exists(masks_path)):
            report.record('data', f'{stream} timestamps are readable', False,
                          f'{times_path} or its mask is missing')
            continue

        times = np.load(times_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        measured = measure_spans(times, masks, chunk=args.span_chunk)
        report.record('data', f'{stream} timestamps are readable', True,
                      f'{times.shape[0]} episodes x {times.shape[1]} timesteps')
        report.note(
            f"{stream} stream: max span {measured['max_span']:,.1f} h, "
            f"median span {measured['median_span']:,.1f} h, "
            f"max step gap {measured['max_step_gap']:,.1f} h, "
            f"min step gap {measured['min_step_gap']}"
        )

        p_min = base_config.get(p_min_key)
        p_max = base_config.get(p_max_key)
        if p_min is None or p_max is None:
            report.record('data', f'{stream} ladder bounds are configured', None,
                          f'{p_min_key}/{p_max_key} are absent, so the inherited 1e4 ladder '
                          f'would be used. That ladder leaves no band phase-coherent past '
                          f'~1e5 h and the revision plan says it should not be run.')
            continue

        # The informative-band criterion is 0.1 <= gap/lambda <= pi, which in periods is
        # 2*gap <= P <= 20*pi*gap ~ 63*gap. P_MAX below 63*Δ_max means the slowest band still
        # wraps at the largest gap the data contains.
        required_p_max = 63.0 * measured['max_span']
        ratio = float(p_max) / required_p_max if required_p_max > 0 else float('inf')
        if 0.5 <= ratio <= 2.0:
            report.record('data', f'{p_max_key} matches the measured span', True,
                          f"configured {float(p_max):.3g} h against 63 x "
                          f"{measured['max_span']:,.0f} h = {required_p_max:.3g} h")
        else:
            report.record(
                'data', f'{p_max_key} matches the measured span', None,
                f"configured {float(p_max):.3g} h, but 63 x the measured max span "
                f"({measured['max_span']:,.0f} h) is {required_p_max:.3g} h -- a factor of "
                f"{ratio:.2g} out.\n"
                f"The configured bound was derived from a max gap of 125,073 h measured on "
                f"the data as it stood BEFORE re-extraction. If the re-extracted arrays "
                f"disagree, update {p_max_key} in the base config before generating the "
                f"sweep: the whole argument for re-spanning rests on this number."
            )

        # P_MIN = 2 x the finest resolution. Below that the fastest band aliases.
        if measured['min_step_gap']:
            required_p_min = 2.0 * measured['min_step_gap']
            if float(p_min) <= required_p_min * 1.5:
                report.record('data', f'{p_min_key} is at or below the Nyquist bound', True,
                              f"configured {float(p_min):g} h against 2 x "
                              f"{measured['min_step_gap']:g} h = {required_p_min:g} h")
            else:
                report.record('data', f'{p_min_key} is at or below the Nyquist bound', None,
                              f"configured {float(p_min):g} h exceeds 2 x the finest observed "
                              f"step of {measured['min_step_gap']:g} h, so the fastest band "
                              f"cannot resolve the closest pair of records.")


def parse_peak_memory(text):
    """Pull the peak VRAM figures out of a run's stdout.

    Args:
        text: Captured stdout.

    Returns:
        The largest peak reported, in GB, or None if none was printed.
    """
    peaks = [float(m) for m in re.findall(r'Rank \d+:\s+([\d.]+)\s+GB', text)]
    return max(peaks) if peaks else None


def seconds_until_job_ends():
    """Seconds left in the SLURM allocation, or None when not running under SLURM.

    SLURM exports `SLURM_JOB_END_TIME` as a Unix timestamp. A job cut off at its limit never
    prints the summary or the `--time` recommendation, which are the outputs of the run, so
    the trial loops use this to stop while there is still time to report.
    """
    raw = os.environ.get('SLURM_JOB_END_TIME')
    if not raw:
        return None
    try:
        return float(raw) - time.time()
    except ValueError:
        return None


def budget_allows(next_estimate, reserve_seconds):
    """Whether another trial of `next_estimate` seconds fits before the deadline.

    Args:
        next_estimate: Expected seconds for the trial about to start, or None if unknown.
        reserve_seconds: Seconds to keep back for the reporting and selection stages.

    Returns:
        (True, '') to proceed, or (False, reason) to stop cleanly.
    """
    remaining = seconds_until_job_ends()
    if remaining is None or next_estimate is None:
        return True, ''
    if next_estimate <= remaining - reserve_seconds:
        return True, ''
    return False, (
        f'{remaining / 60:.1f} min left in the allocation, {reserve_seconds / 60:.1f} min '
        f'reserved for reporting, and the next trial is estimated at '
        f'{next_estimate / 60:.1f} min.'
    )


def run_subprocess(command, log_path, env=None):
    """Run a command, tee its output to a log, and return (status, elapsed, text).

    Args:
        command: Argument list.
        log_path: Where to write the captured output.
        env: Environment for the child, or None to inherit.

    Returns:
        A tuple of (exit status, wall seconds, captured stdout+stderr).
    """
    print(f"    $ {' '.join(command)}", flush=True)
    started = time.time()
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )
    elapsed = time.time() - started
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'w') as f_out:
        f_out.write(completed.stdout)
    print(f"    -> status {completed.returncode} in {elapsed:.1f}s, log {log_path}", flush=True)
    return completed.returncode, elapsed, completed.stdout


def run_stage_memory(args, report, work_dir, base_config):
    """Run two batches of pretraining at the real batch size, per arm, and report peak VRAM.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
        work_dir: Scratch directory.
        base_config: The loaded base experiment config.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: memory, at the real batch size")
    print("=" * 70)
    print("Single-GPU-per-run is a premise of Phase 2, not a measurement, until this passes.")
    print("Two changes are supposed to have bought the headroom: the text embedding narrowing")
    print("from a large LLM's width, and the THP restriction to in-stay records.")
    print()

    config_dir = os.path.join(work_dir, 'memtest_configs')
    os.makedirs(config_dir, exist_ok=True)

    for arm in args.arms:
        config = dict(base_config)
        config['EXPERIMENT_NAME'] = f'phase2_memtest_{arm}'
        config['POSITION_ENCODING'] = arm
        config['MODEL_DIR'] = os.path.join(work_dir, 'models')
        config['PRETRAIN_TOTAL_EPOCH'] = 1
        config_path = os.path.join(config_dir, f'memtest_{arm}.yaml')
        with open(config_path, 'w') as f_out:
            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)

        status, elapsed, text = run_subprocess(
            [sys.executable, 'run_experiment.py', args.dataset_config, config_path,
             '--folds', args.fold, '--tasks', 'none',
             '--num_workers', str(args.num_workers), '--mem_test_mode'],
            os.path.join(work_dir, 'logs', f'memtest_{arm}.log')
        )
        peak = parse_peak_memory(text)
        if status != 0:
            report.record('memory', f'{arm} arm runs a batch at batch size '
                                    f'{config["BATCH_SIZE"]}', False,
                          f'exit status {status}; see the log. Last lines:\n'
                          + '\n'.join(text.strip().splitlines()[-15:]))
            continue
        report.record('memory', f'{arm} arm runs a batch at batch size '
                                f'{config["BATCH_SIZE"]}', True)
        if peak is None:
            report.record('memory', f'{arm} arm reports peak VRAM', None,
                          'no peak-memory line in the output')
        else:
            report.note(f"{arm} arm peak VRAM: {peak:.2f} GB at batch "
                        f"{config['BATCH_SIZE']}")
            import torch
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            headroom = peak / total_gb
            report.record(
                'memory', f'{arm} arm fits one GPU with headroom', headroom < 0.8,
                f'{peak:.2f} GB is {headroom:.0%} of the {total_gb:.0f} GB card. Above 80% '
                f'the run is one long episode away from an OOM, and the fix is a smaller '
                f'batch size, not more GPUs -- the sweep gets its parallelism from running '
                f'independent trials, not from splitting one.'
                if headroom >= 0.8 else f'{peak:.2f} GB of {total_gb:.0f} GB'
            )


def write_test_spec(args, work_dir, base_config):
    """Write the miniature sweep's base config and spec into the work directory.

    Args:
        args: Parsed command-line arguments.
        work_dir: Scratch directory.
        base_config: The real base config, copied and shrunk.

    Returns:
        The path of the spec written.
    """
    # Absolute, because every path below is written into the spec and load_spec() resolves a
    # relative path in a spec *relative to that spec's own directory* -- the convention that lets
    # the real phase2_spec.yaml name its base config as a bare filename. The SLURM wrapper
    # defaults WORK_DIR to a relative `log/test_phase2_<jobid>`, so a relative BASE_CONFIG was
    # joined onto the spec dir and doubled. OUTPUT_DIR and MANIFEST carried the same fault and
    # would have failed next.
    spec_dir = os.path.abspath(os.path.join(work_dir, 'spec'))
    os.makedirs(spec_dir, exist_ok=True)

    test_base = dict(base_config)
    test_base['EXPERIMENT_NAME'] = 'phase2_test_base'
    test_base['MODEL_DIR'] = os.path.abspath(os.path.join(work_dir, 'models'))
    test_base['PRETRAIN_TOTAL_EPOCH'] = args.epochs
    test_base['FINETUNE_TOTAL_EPOCH'] = args.epochs
    for name, entry in TEST_GRID.items():
        test_base[name] = entry['values'][0]
    base_path = os.path.join(spec_dir, 'phase2_test_base.yaml')
    with open(base_path, 'w') as f_out:
        yaml.dump(test_base, f_out, default_flow_style=False, sort_keys=False)

    spec = {
        'SPEC_NAME': 'phase2test',
        'BASE_CONFIG': base_path,
        'DATASET_CONFIG': os.path.abspath(args.dataset_config),
        'OUTPUT_DIR': os.path.join(spec_dir, 'trials'),
        'MANIFEST': os.path.join(spec_dir, 'phase2test_manifest.yaml'),
        'FOLD': args.fold,
        'ARMS': {arm: {'POSITION_ENCODING': arm} for arm in args.arms},
        'ALIASES': TEST_ALIASES,
        'GRID': TEST_GRID,
    }
    spec_path = os.path.join(spec_dir, 'phase2test_spec.yaml')
    with open(spec_path, 'w') as f_out:
        yaml.dump(spec, f_out, default_flow_style=False, sort_keys=False)
    return spec_path


def run_stage_pipeline(args, report, work_dir, base_config):
    """Run a miniature sweep all the way through, on the real data.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
        work_dir: Scratch directory.
        base_config: The loaded base experiment config.

    Returns:
        Mean seconds per pretraining trial, or None if the stage did not get that far.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: the full pipeline, truncated")
    print("=" * 70)
    print(f"{len(args.arms)} arms x {len(TEST_GRID)} hyperparameters at two values each,")
    print(f"{args.epochs} epochs, at most {args.limit_episodes} episodes per partition.")
    print()

    spec_path = write_test_spec(args, work_dir, base_config)

    status, _, text = run_subprocess(
        [sys.executable, 'generate_tuning_configs.py', spec_path],
        os.path.join(work_dir, 'logs', 'generate.log')
    )
    report.record('pipeline', 'the config generator expands the spec', status == 0,
                  '\n'.join(text.strip().splitlines()[-10:]) if status else '')
    if status != 0:
        return None

    from hp_tuning.spec import finetune_trials, load_manifest, pretrain_trials
    manifest_path = os.path.join(work_dir, 'spec', 'phase2test_manifest.yaml')
    manifest = load_manifest(manifest_path)
    pretrains = pretrain_trials(manifest)
    finetunes = finetune_trials(manifest)
    report.record('pipeline', 'the manifest names the expected trials',
                  len(pretrains) == len(args.arms) * 3 and len(finetunes) == len(args.arms) * 2,
                  f'{len(pretrains)} pretrain, {len(finetunes)} finetune; expected '
                  f'{len(args.arms) * 3} and {len(args.arms) * 2}')

    common = ['--folds', args.fold, '--num_workers', str(args.num_workers),
              '--limit_episodes', str(args.limit_episodes)]

    pretrain_times = []
    finetune_times = []
    ran_out = None
    for index, trial in enumerate(pretrains):
        # Slowest completed trial rather than the mean, so a slow outlier does not overrun
        # the limit and cost the report.
        allowed, reason = budget_allows(
            max(pretrain_times) if pretrain_times else None, args.reserve_seconds)
        if not allowed:
            ran_out = f'pretrain {index + 1}/{len(pretrains)} -- {reason}'
            break
        print(f"\n  pretrain {index + 1}/{len(pretrains)}: {trial['name']}")
        status, elapsed, text = run_subprocess(
            [sys.executable, 'run_experiment.py', manifest['dataset_config'], trial['config'],
             '--tasks', 'none'] + common,
            os.path.join(work_dir, 'logs', f"pretrain_{trial['name']}.log")
        )
        ok = status == 0
        report.record('pipeline', f"pretrain {trial['name']}", ok,
                      '\n'.join(text.strip().splitlines()[-15:]) if not ok else
                      f'{elapsed:.1f}s')
        if ok:
            pretrain_times.append(elapsed)
            encoder_dir = os.path.join(manifest['model_dir'], trial['name'],
                                       manifest['fold'], 'pretrained')
            missing = [name for name in ('value_encoder.pt', 'event_encoder.pt')
                       if not os.path.exists(os.path.join(encoder_dir, name))]
            report.record('pipeline', f"{trial['name']} wrote its encoder weights",
                          not missing,
                          f'missing {missing} in {encoder_dir}. The finetune stage loads these '
                          f'and will fail without them.' if missing else '')
            eval_path = os.path.join(encoder_dir, 'evaluation', 'evaluation_pretrained.yaml')
            report.record('pipeline', f"{trial['name']} wrote its pretraining losses",
                          os.path.exists(eval_path),
                          f'{eval_path} is absent, so learning rate cannot be selected on '
                          f'pretraining loss.' if not os.path.exists(eval_path) else '')

    for index, trial in enumerate(finetunes):
        # A finetune has no completed sibling to estimate from on the first pass, so fall back
        # to the pretrain times; the two stages train the same encoders for the same epochs.
        reference = finetune_times or pretrain_times
        allowed, reason = budget_allows(
            max(reference) if reference else None, args.reserve_seconds)
        if not allowed:
            ran_out = ran_out or f'finetune {index + 1}/{len(finetunes)} -- {reason}'
            break
        print(f"\n  finetune {index + 1}/{len(finetunes)}: {trial['name']}")
        status, elapsed, text = run_subprocess(
            [sys.executable, 'run_experiment.py', manifest['dataset_config'], trial['config'],
             '--tasks', 'mortality'] + common,
            os.path.join(work_dir, 'logs', f"finetune_{trial['name']}.log")
        )
        ok = status == 0
        if ok:
            finetune_times.append(elapsed)
        report.record('pipeline', f"finetune {trial['name']}", ok,
                      '\n'.join(text.strip().splitlines()[-15:]) if not ok else
                      f'{elapsed:.1f}s')
        if ok:
            eval_path = os.path.join(manifest['model_dir'], trial['name'], manifest['fold'],
                                     'mortality', 'evaluation', 'evaluation_mortality.yaml')
            has_scores = False
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f_in:
                    data = yaml.safe_load(f_in)
                has_scores = bool((data or {}).get('validation_scores'))
            report.record('pipeline', f"{trial['name']} wrote mortality validation scores",
                          has_scores,
                          f'{eval_path} has no validation_scores block, so the three '
                          f'downstream-selected hyperparameters have nothing to rank on.'
                          if not has_scores else '')

    if ran_out is not None:
        # Per-trial cost is close to linear in the episode count, so the fraction of trials
        # that fitted gives the shrink factor, less 20% for margin.
        n_trials = len(pretrains) + len(finetunes)
        n_done = len(pretrain_times) + len(finetune_times)
        suggestion = ''
        if n_done:
            fraction = n_done / n_trials
            suggested = max(25, int(args.limit_episodes * fraction * 0.8))
            suggestion = (
                f' {n_done} of {n_trials} trials finished, so about {fraction:.0%} of the '
                f'sweep fitted the allocation. Re-run with '
                f'TEST_ARGS="--limit_episodes {suggested}" to fit the current limit, or '
                f'raise --time so the whole sweep runs -- a request is capped at 7 days and '
                f'costs nothing so long as the job uses what it asks for.'
            )
        report.record('pipeline', 'the sweep finished inside the allocation', None,
                      f'Stopped before {ran_out}{suggestion}')

    status, _, text = run_subprocess(
        [sys.executable, 'report_tuning_results.py', manifest_path,
         '--tables_dir', os.path.join(work_dir, 'tables')],
        os.path.join(work_dir, 'logs', 'report.log')
    )
    report.record('pipeline', 'the reporting script runs', status == 0,
                  '\n'.join(text.strip().splitlines()[-15:]) if status else '')
    for name in ('phase2test_tuning.docx', 'phase2test_tuning.csv'):
        path = os.path.join(work_dir, 'tables', name)
        report.record('pipeline', f'the report wrote {name}', os.path.exists(path), path)

    status, _, text = run_subprocess(
        [sys.executable, 'select_tuned_hyperparameters.py', manifest_path,
         '--output', os.path.join(work_dir, 'assembled.yaml')],
        os.path.join(work_dir, 'logs', 'select.log')
    )
    report.record('pipeline', 'the selection script runs and assembles a config', status == 0,
                  '\n'.join(text.strip().splitlines()[-15:]) if status else '')
    if status == 0:
        with open(os.path.join(work_dir, 'assembled.yaml'), 'r') as f_in:
            assembled = yaml.safe_load(f_in)
        missing = [name for name in TEST_GRID if name not in assembled]
        report.record('pipeline', 'the assembled config carries every tuned value',
                      not missing, f'missing {missing}')

    return sum(pretrain_times) / len(pretrain_times) if pretrain_times else None


EPOCH_LINE = re.compile(
    rf'^{EPOCH_TIMING_PREFIX}\s+n=(\d+)\s+mean=([\d.]+)\s+first=([\d.]+)', re.MULTILINE)
STARTUP_LINE = re.compile(rf'^{STARTUP_TIMING_PREFIX}\s+([\d.]+)', re.MULTILINE)
# A resumed run that is already at its epoch budget completes no epochs and measures nothing.
ZERO_EPOCH_LINE = re.compile(rf'^{EPOCH_TIMING_PREFIX}\s+n=0\s*$', re.MULTILINE)


def run_stage_timing(args, report, work_dir, base_config, trial_estimate=None):
    """Time full-size epochs directly, one short run per arm.

    Extrapolating from the truncated pipeline stage cannot work: a trial's wall time is fixed
    startup and teardown plus per-epoch cost, and dividing the whole by the epoch count then
    scaling by the episode ratio multiplies the fixed part by both factors. At 2 epochs over
    400 of 19,112 episodes that is a factor of about 4,800 on every startup second, which is
    how a 0.6 h trial was once reported as 651 h.

    Epochs at full size are cheap enough to measure outright, so this runs a few of them and
    reads the per-epoch cost off `pretrain_model` directly. `--limit_episodes` is deliberately
    absent.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
        work_dir: Scratch directory.
        base_config: The real base config, copied per arm.
        trial_estimate: Seconds one `run_experiment.py` invocation took in the pipeline stage,
            used as the budget estimate for the first arm. Without it the first run would be
            waved through with any amount of time left and killed mid-epoch, losing the report.

    Returns:
        {arm: {'startup': seconds, 'epoch': seconds, 'epochs': n}} for every arm that finished.
    """
    print("\n" + "=" * 70)
    print("STAGE 5: per-epoch cost at full size")
    print("=" * 70)
    print(f"{args.timing_epochs} epochs over every episode in the fold, no truncation. The "
          f"first")
    print("epoch is discarded: it pays worker startup, cuDNN autotuning and allocator growth.")

    import torch

    if not torch.cuda.is_available():
        report.note('No per-epoch timing: no GPU.')
        return {}

    timing_dir = os.path.abspath(os.path.join(work_dir, 'timing'))
    os.makedirs(timing_dir, exist_ok=True)
    timings = {}

    for arm in args.arms:
        config = dict(base_config)
        config['EXPERIMENT_NAME'] = f'phase2_timing_{arm}'
        config['POSITION_ENCODING'] = arm
        config['MODEL_DIR'] = os.path.join(timing_dir, 'models')
        config['PRETRAIN_TOTAL_EPOCH'] = args.timing_epochs
        config_path = os.path.join(timing_dir, f'timing_{arm}.yaml')
        with open(config_path, 'w') as f_out:
            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)

        # run_experiment.py puts checkpoints under ./checkpoints/<EXPERIMENT_NAME>/, outside
        # work_dir, and pretrain_model resumes from them. A run that completes cleans up after
        # itself, but one killed at the allocation limit does not -- and the next run would
        # resume at the final epoch, complete zero epochs, and report no per-epoch cost at all.
        stale = os.path.join('checkpoints', config['EXPERIMENT_NAME'], args.fold, 'pretrained')
        if os.path.isdir(stale):
            print(f"    removing a stale checkpoint at {stale}")
            shutil.rmtree(stale)

        # One arm's measurement is enough to recommend from, so a tight allocation should stop
        # here rather than lose the run mid-epoch and report nothing. Before any arm has been
        # measured, the pipeline stage's per-invocation wall time is the closest proxy there is.
        if timings:
            estimate = max(t['startup'] + t['epoch'] * args.timing_epochs
                           for t in timings.values()) * 1.5
        else:
            estimate = trial_estimate
        allowed, reason = budget_allows(estimate, args.reserve_seconds)
        if not allowed:
            report.record('timing', f'{arm}: per-epoch cost measured', None,
                          f'Skipped: {reason}')
            continue

        print(f"\n  {arm}: {args.timing_epochs} full-size epochs")
        status, elapsed, text = run_subprocess(
            [sys.executable, 'run_experiment.py', os.path.abspath(args.dataset_config),
             config_path, '--folds', args.fold, '--tasks', 'none',
             '--num_workers', str(args.num_workers)],
            os.path.join(work_dir, 'logs', f'timing_{arm}.log')
        )
        epoch_match = EPOCH_LINE.search(text)
        startup_match = STARTUP_LINE.search(text)
        ran_no_epochs = ZERO_EPOCH_LINE.search(text) is not None
        ok = status == 0 and epoch_match is not None and startup_match is not None
        if status:
            detail = '\n'.join(text.strip().splitlines()[-15:])
        elif ran_no_epochs:
            detail = (f'the run completed zero epochs, so it measured nothing. It resumed from '
                      f'a checkpoint at or past {args.timing_epochs} epochs -- check '
                      f'checkpoints/{config["EXPERIMENT_NAME"]}/ and remove it.')
        else:
            detail = (f'the run finished but printed no {EPOCH_TIMING_PREFIX} / '
                      f'{STARTUP_TIMING_PREFIX} line, so there is nothing to size the '
                      f'request from')
        report.record('timing', f'{arm}: per-epoch cost measured', ok, detail)
        if not ok:
            continue

        timings[arm] = {
            'startup': float(startup_match.group(1)),
            'epoch': float(epoch_match.group(2)),
            'epochs': int(epoch_match.group(1)),
            'first_epoch': float(epoch_match.group(3)),
            'wall': elapsed,
        }
        entry = timings[arm]
        print(f"    startup {entry['startup']:.1f}s, first epoch "
              f"{entry['first_epoch']:.1f}s, steady {entry['epoch']:.1f}s/epoch")
        report.note(
            f"{arm}: {entry['epoch']:.1f}s per full-size epoch, {entry['startup']:.1f}s "
            f"startup, {entry['first_epoch']:.1f}s first epoch."
        )

    return timings


def report_timing(args, report, timings, base_config):
    """Print the --time to request for each array, from the measured per-epoch cost.

    Args:
        args: Parsed command-line arguments.
        report: The Report to record into.
        timings: The mapping returned by run_stage_timing.
        base_config: The loaded base experiment config.
    """
    print("\n" + "=" * 70)
    print("STAGE 6: what to request for the real jobs")
    print("=" * 70)

    if not timings:
        report.note('No --time recommendation: no arm produced a per-epoch measurement.')
        return

    pretrain_epochs = base_config.get('PRETRAIN_TOTAL_EPOCH', 200)
    finetune_epochs = base_config.get('FINETUNE_TOTAL_EPOCH', 500)

    # The array shares one --time across its tasks, so the slowest arm sets it.
    arm, entry = max(timings.items(),
                     key=lambda kv: kv[1]['startup'] + kv[1]['epoch'] * pretrain_epochs)
    pretrain_seconds = entry['startup'] + entry['epoch'] * pretrain_epochs
    finetune_seconds = entry['startup'] + entry['epoch'] * finetune_epochs

    def hours(seconds, buffer=1.5):
        """Buffered seconds, rounded up to a whole hour, never below one."""
        return max(1, int(seconds * buffer / 3600) + 1)

    pretrain_hours = hours(pretrain_seconds)
    finetune_hours = hours(finetune_seconds)

    report.note(
        f'Full pretrain trial, {arm} (the slower arm): {pretrain_seconds / 3600:.2f} h for '
        f'{pretrain_epochs} epochs at {entry["epoch"]:.1f}s each, plus '
        f'{entry["startup"]:.1f}s startup.'
    )

    print()
    print(f"  Measured on the {arm} arm, which is the slower of those run.")
    print(f"  Pretrain: {pretrain_epochs} epochs x {entry['epoch']:.1f}s + "
          f"{entry['startup']:.1f}s startup = {pretrain_seconds / 3600:.2f} h")
    print(f"  Finetune: {finetune_epochs} epochs, priced at the pretrain per-epoch cost, "
          f"= {finetune_seconds / 3600:.2f} h")
    print()
    print(f"  sbatch --time={pretrain_hours:02d}:00:00 --array=0-21%8 \\")
    print(f"      SLURM/slurm_tune_pretrain.sh <manifest>")
    print(f"  sbatch --time={finetune_hours:02d}:00:00 --array=0-13%8 \\")
    print(f"      SLURM/slurm_tune_finetune.sh <manifest>")
    print()
    print("  Both figures carry a 50% buffer and round up to the hour. The finetune number is")
    print("  an upper bound: finetuning drops the generator, discriminator and THP, so its")
    print("  epochs are cheaper than the pretrain epochs it is priced at here. Early stopping")
    print("  fires after 30 epochs without improvement, so most trials finish well short of")
    print("  either ceiling.")
    report.note(f'Recommended --time: pretrain {pretrain_hours:02d}:00:00, '
                f'finetune {finetune_hours:02d}:00:00.')


def main(argv=None):
    """Run the smoke test.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status: 0 if nothing failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dataset_config', type=str,
                        default='TransEHR2/configs/datasets/mimic4.yaml')
    parser.add_argument('--base_config', type=str,
                        default='TransEHR2/configs/experiments/tuning/phase2_base.yaml')
    parser.add_argument('--fold', type=str, default='fold0')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Scratch directory. Everything the test writes goes here, '
                             'including a MODEL_DIR override, so the real models/ tree is '
                             'never touched.')
    parser.add_argument('--arms', type=str, nargs='+', default=['additive', 'rope'])
    parser.add_argument('--reserve_seconds', type=int, default=420,
                        help='Seconds held back from the SLURM allocation for the reporting '
                             'and selection stages. The trial loops stop rather than start a '
                             'trial that would not finish inside the remainder, so the run '
                             'still prints its summary and its --time recommendation instead '
                             'of being cut off mid-trial.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Epochs per trial in the pipeline stage. Two is the minimum that '
                             'exercises the improvement check and the scheduler step.')
    parser.add_argument('--limit_episodes', type=int, default=400,
                        help='Episodes per partition in the pipeline stage')
    parser.add_argument('--timing_epochs', type=int, default=3,
                        help='Full-size epochs to run per arm in the timing stage. The first '
                             'is discarded, so three gives two steady-state samples.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--span_chunk', type=int, default=4096,
                        help='Episodes per block when measuring temporal spans')
    parser.add_argument('--only', type=str, nargs='+', choices=STAGES, default=None,
                        help='Run only these stages')
    parser.add_argument('--skip', type=str, nargs='+', choices=STAGES, default=[],
                        help='Skip these stages')
    parser.add_argument('--keep_work_dir', action='store_true',
                        help='Do not warn about leftover files in the work directory')
    args = parser.parse_args(argv)

    stages = list(args.only) if args.only else [s for s in STAGES if s not in args.skip]

    # Everything derived from work_dir ends up in the generated spec, and load_spec() resolves a
    # relative path in a spec *relative to that spec's own directory* -- the convention that lets
    # the real phase2_spec.yaml name its base config as a bare filename. The SLURM wrapper
    # defaults WORK_DIR to a relative `log/test_phase2_<jobid>`, so a relative BASE_CONFIG got
    # joined onto the spec dir and doubled the path. Absolute here, once, and every path the spec
    # carries is unambiguous.
    args.work_dir = os.path.abspath(args.work_dir)

    print("=" * 70)
    print("Phase 2 pipeline smoke test")
    print("=" * 70)
    print(f"  dataset config: {args.dataset_config}")
    print(f"  base config:    {args.base_config}")
    print(f"  fold:           {args.fold}")
    print(f"  work dir:       {args.work_dir}")
    print(f"  arms:           {', '.join(args.arms)}")
    print(f"  stages:         {', '.join(stages)}")

    os.makedirs(args.work_dir, exist_ok=True)
    with open(args.base_config, 'r') as f_in:
        base_config = yaml.safe_load(f_in)

    report = Report()

    if 'environment' in stages:
        run_stage_environment(args, report)
    if 'data' in stages:
        run_stage_data(args, report, base_config)
    if 'memory' in stages:
        run_stage_memory(args, report, args.work_dir, base_config)

    mean_trial_seconds = None
    if 'pipeline' in stages:
        mean_trial_seconds = run_stage_pipeline(args, report, args.work_dir, base_config)
        if mean_trial_seconds is not None:
            report.note(f'Truncated pipeline trial: {mean_trial_seconds:.1f}s of whole-process '
                        f'wall time for {args.epochs} epochs over at most '
                        f'{args.limit_episodes} episodes. Diagnostic only -- the --time '
                        f'recommendation comes from the timing stage, not from this.')

    # Its own stage, so `--only timing` can re-measure without re-running the sweep.
    if 'timing' in stages:
        timings = run_stage_timing(args, report, args.work_dir, base_config,
                                   trial_estimate=mean_trial_seconds)
        report_timing(args, report, timings, base_config)

    status = report.summarise()
    print(f"\nEverything the test wrote is under {args.work_dir}")
    return status


if __name__ == '__main__':
    sys.exit(main())
