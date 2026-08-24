"""Reads the per-patient artifacts Stage B writes (sections 2.5-2.7, 5).

Readings this module commits to
-------------------------------

* **The reader is indexed by patient, not by episode** (section 4.1). One
  ``timeseries.csv`` read and parse serves every episode of that patient,
  and a patient's episodes differ only in where their window ends. The
  insertion pass puts the results back in ``labels.csv`` order, so the
  canonical order survives the reordering that parallelism imposes.
* **``labels.csv`` supplies the row order and the targets; ``stays.csv``
  supplies ``INDEX_TIME``** (section 3, B4). The file's own row order is
  canonical -- patient directories lexicographically, then ``STAY_INDEX``
  -- so ``time_to_event.npy``, ``event_type.npy`` and ``episode_ids.pkl``
  are columns of it rather than a second derivation. ``INDEX_TIME`` is the
  one thing section 3 leaves out of ``labels.csv``, so it is looked up in
  ``stays.csv`` on ``(PATID, STAY_INDEX)`` -- in labels order, never in
  stays order.
* **Categorical, ordinal and text columns are read as strings** (section
  5's dtype trap). ``category_map`` is keyed on the YAML's values, and
  ``BLDUA``'s declared levels are numeric-looking *strings* (``"0"``,
  ``"1-24"``, ...). A patient whose ``BLDUA`` column happens to hold only
  ``0`` and blanks is read by pandas as a numeric column, and
  ``0.0 in {"0": 0, ...}`` is ``False`` -- a data-dependent, per-patient,
  silent miss. Forcing the dtype is half the fix; ``str(value)`` at the
  lookup is the other half (see ``preprocessing``).
* **The empty string is the only missing marker.** Stage B writes every
  CSV with ``na_rep=""``, so ``keep_default_na`` is switched off and
  ``na_values`` is ``['']``. Pandas' default NA strings include ``'None'``,
  which is ``UBAC``'s declared level 0 -- urine bacteria, none seen --
  so under the defaults that level is unobservable, its indicator 0 on
  every timestep it was actually measured. It is the same family as the
  ``BLDUA`` trap above and the only declared level currently exposed:
  scanning every ``categorical`` and ``ordinal`` ``category_map`` against
  ``pandas._libs.parsers.STR_NA_VALUES`` returns ``UBAC`` alone.
* **Rows are not dropped from the value stream.** Section 2.6 writes one
  row per distinct minute, and a row carrying only text or only a
  dispensation is still a timestep -- section 2.7 says so explicitly, and
  invariant 4 requires every ``drugs.csv`` timestamp to name a
  ``timeseries.csv`` row. Only the *event* stream is thinned to the rows
  that carry an event, which is what makes it a separate stream.
* **``DRUG_FEATS`` has no timeseries column** (section 2.6), so drugs reach
  the extractor through ``drugs.csv`` alone, joined on ``TIMESTAMP``. The
  reader therefore hands back text columns and drug rows separately even
  though section 4.3 stores them as one family; the asymmetry is in the
  source, not in the storage.
"""

import numpy as np
import os
import pandas as pd
import re

from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Tuple

# Written by IBDdataprep's build_labels.py (section 3).
LABELS_COLUMNS = ('PATID', 'STAY_INDEX', 'TIME_TO_EVENT', 'EVENT_TYPE')

TIMESTAMP_COLUMN = 'TIMESTAMP'
STAYS_FILE = 'stays.csv'
TIMESERIES_FILE = 'timeseries.csv'
DRUGS_FILE = 'drugs.csv'


class EHRDataReader(Sequence):
    """Reads ``{root_dir}/{PATID}/`` for one patient at a time.

    Indexing the reader yields everything needed to extract every episode
    of one patient: the episode rows in canonical order, the patient's
    whole timeseries, and their dispensations.
    """

    def __init__(
        self,
        labels_path: str,
        root_dir: str,
        valued_feats: List[str],
        event_feats: List[str],
        text_feats: Optional[List[str]] = None,
        drug_feats: Optional[List[str]] = None,
        static_feats: Optional[List[str]] = None,
        string_feats: Optional[List[str]] = None,
        n_examples: Optional[int] = None
    ):
        """Initialize the reader.

        Args:
            labels_path: ``labels.csv`` from ``build_labels.py``. Its row
                order is the canonical episode order (section 3).
            root_dir: Directory holding one subdirectory per ``PATID``.
            valued_feats: Value-associated feature names.
            event_feats: Event-associated feature names.
            text_feats: Text feature names (may be None or empty).
            drug_feats: Drug feature names. Carried for the lookup family
                (section 4.3), not used to select columns: section 2.6
                gives ``DRUG_FEATS`` no ``timeseries.csv`` column, so the
                data comes from ``drugs.csv``.
            static_feats: Static feature names. ``STATIC_FEATS`` is empty
                under section A.3; the path is kept, not exercised.
            string_feats: Feature names whose column must be read as
                strings -- every categorical and ordinal feature, plus the
                text features. See the dtype trap above.
            n_examples: Read only the first N rows of ``labels.csv``, for
                debugging. Episodes, not patients: the patient set follows
                from the rows kept.
        """

        super().__init__()

        self.labels_path = labels_path
        self.root_dir = str(root_dir)
        self.valued_feats = valued_feats
        self.event_feats = event_feats
        self.text_feats = text_feats or []
        self.drug_feats = drug_feats or []
        self.static_feats = static_feats or []
        self.string_feats = string_feats or []

        labels = pd.read_csv(
            labels_path,
            nrows=n_examples,
            dtype={'PATID': int, 'STAY_INDEX': int, 'EVENT_TYPE': int},
        )
        missing = [c for c in LABELS_COLUMNS if c not in labels.columns]
        if missing:
            raise ValueError(
                f"{labels_path} is missing column(s) {missing}; it should "
                f"be the output of IBDdataprep's build_labels.py"
            )
        # The row number *is* the output row. Recorded now, because the
        # per-patient grouping below reorders nothing but must not be
        # mistaken for the canonical order.
        labels = labels.reset_index(drop=True)
        labels['ROW'] = np.arange(len(labels), dtype=np.int64)
        self.labels = labels

        # First-appearance order, which is lexicographic by construction of
        # labels.csv. Used only to schedule work; output order is 'ROW'.
        self.patient_ids = list(dict.fromkeys(labels['PATID'].tolist()))
        self._episodes_by_patient = {
            patid: frame for patid, frame in labels.groupby('PATID')
        }

        self._validate_patient_ids()

    def __len__(self) -> int:
        """Return the number of patients."""
        return len(self.patient_ids)

    @property
    def n_episodes(self) -> int:
        """Number of rows in ``labels.csv``, i.e. output rows."""
        return len(self.labels)

    def patient_dir(self, patid: int) -> Path:
        return Path(self.root_dir) / str(patid)

    def get_episodes(self, patid: int) -> pd.DataFrame:
        """Episode rows for one patient, with ``INDEX_TIME`` joined on.

        Returns the ``labels.csv`` rows for the patient -- in labels order,
        carrying ``ROW`` -- with ``INDEX_TIME`` taken from ``stays.csv`` on
        ``STAY_INDEX``. A labels row naming a stay the patient does not
        have is an error rather than a dropped episode: the fold row
        indices are positions in this file, so the row count cannot move.
        """
        episodes = self._episodes_by_patient[patid]
        stays = pd.read_csv(
            self.patient_dir(patid) / STAYS_FILE,
            usecols=['STAY_INDEX', 'INDEX_TIME'],
            dtype={'STAY_INDEX': int},
        )
        index_times = pd.to_datetime(
            stays['INDEX_TIME'], utc=True, format='ISO8601'
        )
        lookup = dict(zip(stays['STAY_INDEX'], index_times))

        missing = sorted(set(episodes['STAY_INDEX']) - set(lookup))
        if missing:
            raise ValueError(
                f"patient {patid}: {STAYS_FILE} has no STAY_INDEX "
                f"{missing}, which labels.csv names"
            )
        episodes = episodes.copy()
        episodes['INDEX_TIME'] = [lookup[i] for i in episodes['STAY_INDEX']]
        return episodes

    def get_feature_data(
        self, patid: int
    ) -> Tuple[Optional[pd.Series], pd.DataFrame, pd.DataFrame,
               pd.DataFrame, pd.DataFrame]:
        """Read one patient's whole record.

        Returns:
            ``(static_data, val_data, event_data, text_data, drug_data)``.
            The three timeseries frames are indexed by absolute UTC
            timestamp; ``drug_data`` carries ``TIMESTAMP`` as a column
            because it has several rows per timestamp (one per slot).
        """

        ts_path = self.patient_dir(patid) / TIMESERIES_FILE
        header = pd.read_csv(ts_path, nrows=0).columns.tolist()
        dtypes = {
            col: str for col in header
            if col in set(self.string_feats)
        }
        # The empty string is the *only* missing marker Stage B writes
        # (``to_csv(..., na_rep="")``), so pandas' default NA strings are
        # switched off: 'None' is UBAC's declared level 0 and would
        # otherwise be read as missing, making that level unobservable.
        timeseries = pd.read_csv(
            ts_path, dtype=dtypes, keep_default_na=False, na_values=['']
        )
        timeseries[TIMESTAMP_COLUMN] = pd.to_datetime(
            timeseries[TIMESTAMP_COLUMN], utc=True, format='ISO8601'
        )
        timeseries = timeseries.set_index(TIMESTAMP_COLUMN).sort_index()

        val_cols = self._get_feature_column_names(
            self.valued_feats, timeseries
        )
        val_data = timeseries.loc[:, val_cols]

        event_cols = self._get_feature_column_names(
            self.event_feats, timeseries
        )
        event_data = timeseries.loc[:, event_cols].dropna(how='all')

        text_cols = self._get_feature_column_names(
            self.text_feats, timeseries
        )
        text_data = timeseries.loc[:, text_cols]

        drug_data = self._read_drugs(patid)

        if self.static_feats:
            stays = pd.read_csv(self.patient_dir(patid) / STAYS_FILE)
            static_cols = self._get_feature_column_names(
                self.static_feats, stays
            )
            static_data = stays.loc[:, static_cols].iloc[0]
        else:
            static_data = None

        return static_data, val_data, event_data, text_data, drug_data

    def _read_drugs(self, patid: int) -> pd.DataFrame:
        """Read ``drugs.csv``, dropping the over-cap rows.

        Section 2.7 keeps rows the 30-slot cap dropped, marked
        ``SLOT = -1``, so the webapp can show them; the extractor ignores
        them. An absent file means the patient was dispensed nothing.
        """
        path = self.patient_dir(patid) / DRUGS_FILE
        columns = ['TIMESTAMP', 'SLOT', 'CLINVEC_INDEX', 'REL_DAILY_QTY']
        if not path.exists():
            return pd.DataFrame(
                {c: pd.Series(dtype='float64') for c in columns}
            )
        drugs = pd.read_csv(path, usecols=columns)
        drugs = drugs.loc[drugs['SLOT'] >= 0]
        drugs['TIMESTAMP'] = pd.to_datetime(
            drugs['TIMESTAMP'], utc=True, format='ISO8601'
        )
        return drugs

    def _get_feature_column_names(
        self, feature_names: List[str], df: pd.DataFrame
    ) -> List[str]:
        """Get the columns of ``df`` belonging to the named features.

        A vector-valued feature has one column per dimension, named
        ``feature_0``, ``feature_1``, ...; a scalar feature's column is the
        feature name alone.
        """

        feature_columns = []
        for base_name in feature_names:
            matching_columns = [
                col for col in df.columns
                if re.search(rf'^{re.escape(base_name)}(_\d+)?$', col)
            ]
            if matching_columns:
                feature_columns.extend(matching_columns)
        return feature_columns

    def _validate_patient_ids(self) -> None:
        """Check that every patient in ``labels.csv`` has a directory."""
        missing = [
            patid for patid in self.patient_ids
            if not os.path.isdir(os.path.join(self.root_dir, str(patid)))
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} patient(s) named in {self.labels_path} "
                f"have no directory under {self.root_dir}: "
                f"{missing[:10]}"
            )

    def __getitem__(self, index: int):
        """Return one patient's episodes and record.

        Returns:
            ``(patid, episodes, static_data, val_data, event_data,
            text_data, drug_data)``.
        """
        patid = self.patient_ids[index]
        episodes = self.get_episodes(patid)
        (static_data, val_data, event_data,
         text_data, drug_data) = self.get_feature_data(patid)
        return (patid, episodes, static_data, val_data, event_data,
                text_data, drug_data)
