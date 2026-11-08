"""Quarter-based alignment without interpolation or row-count assumptions."""

from __future__ import annotations

import pandas as pd


def complete_quarterly_grid(
    frame: pd.DataFrame,
    group_col: str = "province_code",
    quarter_col: str = "quarter",
    start: str | pd.Period | None = None,
    end: str | pd.Period | None = None,
) -> pd.DataFrame:
    """Return one row per group and quarter; preserve missing observations.

    The default range is the common global min/max, not each group's observed
    bounds. No values (including the target) are interpolated or carried forward.
    """
    work = frame.copy()
    if work.empty:
        raise ValueError("Cannot construct a quarterly grid from an empty frame")
    work[quarter_col] = pd.PeriodIndex(work[quarter_col], freq="Q")
    keys = [group_col, quarter_col]
    if work[keys].isna().any().any() or work.duplicated(keys).any():
        raise ValueError("Quarterly keys must be unique and nonmissing")
    first = pd.Period(start, freq="Q") if start is not None else work[quarter_col].min()
    last = pd.Period(end, freq="Q") if end is not None else work[quarter_col].max()
    if first > last:
        raise ValueError("Quarterly grid start exceeds end")
    grid = pd.MultiIndex.from_product(
        [sorted(work[group_col].unique()), pd.period_range(first, last, freq="Q")],
        names=keys,
    )
    return work.set_index(keys).reindex(grid).reset_index()


def calendar_shift(
    frame: pd.DataFrame,
    value_col: str,
    periods: int,
    group_col: str = "province_code",
    quarter_col: str = "quarter",
) -> pd.Series:
    """Align value at t-periods with t (positive=lag; negative=future label).

    Missing calendar quarters remain missing even on an irregular input frame.
    The caller's row order and index are preserved.
    """
    periods = int(periods)
    quarters = pd.PeriodIndex(frame[quarter_col], freq="Q")
    keys = pd.MultiIndex.from_arrays([frame[group_col], quarters])
    if keys.has_duplicates or frame[group_col].isna().any() or quarters.isna().any():
        raise ValueError("Quarterly keys must be unique and nonmissing")
    lookup = pd.Series(frame[value_col].to_numpy(), index=keys)
    requested = pd.MultiIndex.from_arrays([frame[group_col], quarters - periods])
    return pd.Series(lookup.reindex(requested).to_numpy(), index=frame.index, name=value_col)
