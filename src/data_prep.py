import pandas as pd
import numpy as np
from typing import Tuple, List

DEFAULT_SOHTHRESH = 0.7  

def load_raw(path: str) -> pd.DataFrame:
    """
    Load dataset. Expect columns including:
    'cell' or 'battery' (e.g., B0005), 'cycle', 'charge_current', 'discharge_current',
    'charge_voltage', 'discharge_voltage', 'charge_temp', 'discharge_temp',
    'capacity' (BCt), 'soh' (SOH), maybe 'rul'.
    """
    df = pd.read_csv(path)
    # normalize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    # canonical names
    if 'battery' in df.columns and 'cell' not in df.columns:
        df = df.rename(columns={'battery': 'cell'})
    # try to detect capacity/soh names
    if 'bc' in df.columns and 'capacity' not in df.columns:
        df = df.rename(columns={'bc': 'capacity'})
    return df

def soh_to_rul_for_cell(group: pd.DataFrame, soh_col='soh', cycle_col='cycle', soh_threshold=DEFAULT_SOHTHRESH):
    """
    Given group sorted by cycle, compute rul. If failure not observed, extrapolate linearly.
    """
    grp = group.sort_values(cycle_col).copy()
    cycles = grp[cycle_col].values
    sohs = grp[soh_col].values

    rul = np.full_like(cycles, fill_value=np.nan, dtype=float)

    # find first index where soh <= threshold
    below = np.where(sohs <= soh_threshold)[0]
    if below.size > 0:
        fail_idx = below[0]
        fail_cycle = cycles[fail_idx]
        rul = fail_cycle - cycles
        rul[rul < 0] = 0.0
    else:
        # no failure observed: linear fit on last k points
        k = min(6, len(cycles))
        if k < 3:
            # fallback: large RUL
            rul = np.full_like(cycles, fill_value=1000.0, dtype=float)
        else:
            x = cycles[-k:]
            y = sohs[-k:]
            slope, intercept = np.polyfit(x, y, 1)
            if slope >= 0:
                rul = np.full_like(cycles, fill_value=1000.0, dtype=float)
            else:
                last_cycle = cycles[-1]
                last_soh = sohs[-1]
                est_cycles_to_threshold = (last_soh - soh_threshold) / (-slope)
                fail_cycle_est = last_cycle + est_cycles_to_threshold
                rul = fail_cycle_est - cycles
                rul = np.maximum(rul, 0.0)
    grp = grp.copy()
    grp['rul'] = rul
    return grp

def compute_per_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features per cycle for each cell.
    """
    df = df.copy()
    df = df.sort_values(['cell', 'cycle'])
    # basic numeric columns to consider
    numeric_candidates = [
        'charge_current', 'discharge_current',
        'charge_voltage', 'discharge_voltage',
        'charge_temp', 'discharge_temp',
        'capacity', 'soh'
    ]
    present_nums = [c for c in numeric_candidates if c in df.columns]

    # create power-ish features if voltage*current present
    if 'charge_voltage' in df.columns and 'charge_current' in df.columns:
        df['charge_power'] = df['charge_voltage'] * df['charge_current']
    if 'discharge_voltage' in df.columns and 'discharge_current' in df.columns:
        df['discharge_power'] = df['discharge_voltage'] * df['discharge_current']

    # combine temps
    if 'charge_temp' in df.columns and 'discharge_temp' in df.columns:
        df['temp_mean'] = df[['charge_temp','discharge_temp']].mean(axis=1)
    elif 'charge_temp' in df.columns:
        df['temp_mean'] = df['charge_temp']
    elif 'discharge_temp' in df.columns:
        df['temp_mean'] = df['discharge_temp']

    # rolling features for each cell
    window = 3
    for col in ['capacity', 'soh', 'charge_power', 'discharge_power', 'temp_mean']:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('cell')[col].shift(1).fillna(method='bfill')
            df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
            df[f'{col}_roll_mean3'] = df.groupby('cell')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll_std3'] = df.groupby('cell')[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # cycle based features
    if 'cycle' in df.columns:
        df['cycle_norm'] = df['cycle'] / (df['cycle'].max() + 1)

    # internal resistance approximate if voltage drop and current known:
    # R = (charge_voltage - discharge_voltage) / (charge_current + |discharge_current|)
    if {'charge_voltage','discharge_voltage','charge_current','discharge_current'}.issubset(df.columns):
        denom = df['charge_current'].abs() + df['discharge_current'].abs()
        denom = denom.replace(0, np.nan)
        df['int_res'] = (df['charge_voltage'] - df['discharge_voltage']) / denom
        df['int_res'] = df['int_res'].fillna(method='bfill').fillna(method='ffill').fillna(df['int_res'].median())

    return df

def prepare_dataset(path: str, soh_threshold=DEFAULT_SOHTHRESH, drop_cols: List[str]=None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline:
    - load, normalize
    - compute rul from soh if missing
    - add features
    - return X, y
    """
    df = load_raw(path)

    # ensure cell column
    if 'cell' not in df.columns:
        # try 'id' or 'battery'
        if 'id' in df.columns:
            df = df.rename(columns={'id':'cell'})
        else:
            df['cell'] = 'B000X'

    # compute RUL if missing
    if 'rul' not in df.columns:
        out = []
        for _, group in df.groupby('cell'):
            out.append(soh_to_rul_for_cell(group, soh_col='soh', cycle_col='cycle', soh_threshold=soh_threshold))
        df = pd.concat(out, axis=0).sort_values(['cell','cycle']).reset_index(drop=True)

    # feature engineering
    df = compute_per_cycle_features(df)

    # drop non-feature columns; keep metadata cell & cycle for evaluation
    meta_cols = ['cell','cycle']
    target_col = 'rul'
    exclude = meta_cols + [target_col]
    if drop_cols:
        exclude += drop_cols

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0.0)
    y = df[target_col].astype(float)
    meta = df[meta_cols].copy()
    return X, y, meta