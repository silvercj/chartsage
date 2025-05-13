import numpy as np
import pandas as pd

# --- Derived Field Handlers ---
def compute_ratio(df, num_col, denom_col):
    if num_col not in df.columns or denom_col not in df.columns:
        return None
    # Ensure numeric for ratio, coerce errors
    df_copy = df[[num_col, denom_col]].copy()
    df_copy[num_col] = pd.to_numeric(df_copy[num_col], errors='coerce')
    df_copy[denom_col] = pd.to_numeric(df_copy[denom_col], errors='coerce')
    
    ratio = df_copy[num_col] / df_copy[denom_col].replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()

def compute_histogram_bins_and_freqs(df, col, bins=10):
    # This function should ideally be unified with the more robust version in main.py
    # For now, keeping placeholder logic but recommending consolidation.
    if col not in df.columns:
        return None, None
    
    # Attempt to make the column numeric if it's not, coercing errors
    temp_series = df[col]
    if not pd.api.types.is_numeric_dtype(temp_series):
        temp_series = pd.to_numeric(temp_series, errors='coerce')

    values = temp_series.dropna()
    if len(values) == 0:
        return [], []
    
    is_datetime_representation = False # Simplified: main.py handles actual datetimes better
    # Check if original was likely meant to be timestamp converted from datetime by main.py's histogram
    if pd.api.types.is_integer_dtype(values) and values.max() > 1e9: # Heuristic for timestamps in sec
        is_datetime_representation = True

    counts, bin_edges = np.histogram(values, bins='auto')

    if is_datetime_representation:
        bin_labels = [f'{pd.to_datetime(bin_edges[i], unit="s").strftime("%Y-%m-%d")} to {pd.to_datetime(bin_edges[i+1], unit="s").strftime("%Y-%m-%d")}' for i in range(len(bin_edges)-1)]
    else:
        bin_labels = [f'{float(bin_edges[i]):,.2f}-{float(bin_edges[i+1]):,.2f}' for i in range(len(bin_edges)-1)]
    return bin_labels, counts.tolist()

def compute_group_count(df, group_col):
    if group_col not in df.columns:
        return None, None 
    counts = df[group_col].value_counts().sort_index()
    return counts.index.tolist(), counts.values.tolist()

def compute_sum(df, value_col, group_col=None):
    if value_col not in df.columns:
        return None # Match type for ungrouped
    if group_col:
        if group_col not in df.columns:
            return None, None 
        # Ensure value_col is numeric before grouping
        temp_df = df[[group_col, value_col]].copy()
        temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
        # .sum() naturally skips NaNs resulting from coercion
        agg_result = temp_df.groupby(group_col)[value_col].sum().sort_index()
        return agg_result.index.tolist(), agg_result.values.tolist()
    else:
        # Ensure value_col is numeric for scalar sum
        numeric_series = pd.to_numeric(df[value_col], errors='coerce')
        return numeric_series.sum() # Returns scalar, .sum() skips NaNs

def compute_mean(df, value_col, group_col=None):
    if value_col not in df.columns:
        if group_col: 
            return None, None 
        return None 

    if group_col:
        if group_col not in df.columns:
            return None, None 
        temp_df = df[[group_col, value_col]].copy()
        temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
        # .mean() naturally skips NaNs
        agg_result = temp_df.groupby(group_col)[value_col].mean().sort_index()
        return agg_result.index.tolist(), agg_result.values.tolist()
    else:
        numeric_series = pd.to_numeric(df[value_col], errors='coerce')
        return numeric_series.mean() # Returns scalar, .mean() skips NaNs

def compute_median(df, value_col, group_col=None):
    if value_col not in df.columns:
        if group_col:
            return None, None
        return None

    if group_col:
        if group_col not in df.columns:
            return None, None
        temp_df = df[[group_col, value_col]].copy()
        temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
        # .median() naturally skips NaNs
        agg_result = temp_df.groupby(group_col)[value_col].median().sort_index()
        return agg_result.index.tolist(), agg_result.values.tolist()
    else:
        numeric_series = pd.to_numeric(df[value_col], errors='coerce')
        return numeric_series.median() # Returns scalar, .median() skips NaNs

def compute_correlation(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        return None
    
    df_copy = df[[col1, col2]].copy() # Work on a copy
    df_copy[col1] = pd.to_numeric(df_copy[col1], errors='coerce')
    df_copy[col2] = pd.to_numeric(df_copy[col2], errors='coerce')
    
    df_copy.dropna(subset=[col1, col2], inplace=True)
    if df_copy.empty or len(df_copy) < 2: # Correlation requires at least 2 non-NaN pairs
        return None 
    return df_copy[col1].corr(df_copy[col2])

# --- Main Resolver ---
def resolve_derived_field(field_name, df, chart_obj=None):
    # If chart_obj is provided and has a sibling key for this field, use its derived_from
    # This section handles explicit derivations defined in the chart object
    if chart_obj and field_name in chart_obj and isinstance(chart_obj[field_name], dict) and 'derived_from' in chart_obj[field_name]:
        derived_from = chart_obj[field_name]['derived_from']
        
        # Ratio (expects 2 columns in derived_from)
        if field_name.endswith('_ratio') and len(derived_from) == 2:
            return compute_ratio(df, derived_from[0], derived_from[1])
        
        # Histogram bins/frequencies (expects 1 column in derived_from for the source data)
        if field_name.endswith('_bins'):
            bins, _ = compute_histogram_bins_and_freqs(df, derived_from[0]) # Uses updated histogram
            return bins
        if field_name.endswith('_frequencies'):
            _, freqs = compute_histogram_bins_and_freqs(df, derived_from[0]) # Uses updated histogram
            return freqs
            
        # Group count (expects 1 column in derived_from for grouping)
        if field_name.startswith('group_count_by_') and len(derived_from) == 1:
            _, values = compute_group_count(df, derived_from[0])
            return values 

        # Aggregations (sum, mean, median)
        agg_handled = False
        agg_values = None
        result_is_scalar = False

        if field_name.startswith('sum_') and len(derived_from) >= 1:
            value_col = derived_from[0]
            group_col = derived_from[1] if len(derived_from) > 1 else None
            result = compute_sum(df, value_col, group_col=group_col)
            if group_col: _, agg_values = result
            else: agg_values = result; result_is_scalar = True
            agg_handled = True
        elif field_name.startswith('mean_') and len(derived_from) >= 1:
            value_col = derived_from[0]
            group_col = derived_from[1] if len(derived_from) > 1 else None
            result = compute_mean(df, value_col, group_col=group_col)
            if group_col: _, agg_values = result 
            else: agg_values = result; result_is_scalar = True
            agg_handled = True
        elif field_name.startswith('median_') and len(derived_from) >= 1:
            value_col = derived_from[0]
            group_col = derived_from[1] if len(derived_from) > 1 else None
            result = compute_median(df, value_col, group_col=group_col)
            if group_col: _, agg_values = result
            else: agg_values = result; result_is_scalar = True
            agg_handled = True
        
        if agg_handled:
            # For grouped aggregations, agg_values is a list. For ungrouped, it's a scalar.
            # get_field_as_list expects a list for chart data or a scalar if it's a single metric.
            # If it's scalar and meant for a chart axis, it should be wrapped in a list by get_field_as_list.
            return agg_values 

        # Correlation (expects 2 columns in derived_from)
        if field_name.startswith('correlation_') and len(derived_from) == 2:
            return compute_correlation(df, derived_from[0], derived_from[1]) # Returns scalar

    # Fallback to simple pattern matching (primarily for ungrouped or specific patterns)
    if '_to_' in field_name and field_name.endswith('_ratio'):
        parts = field_name.replace('_ratio', '').split('_to_')
        if len(parts) == 2: return compute_ratio(df, parts[0], parts[1])
    
    if field_name.endswith('_bins'):
        base_col = field_name.replace('_bins', '')
        bins, _ = compute_histogram_bins_and_freqs(df, base_col); return bins
    if field_name.endswith('_frequencies'):
        base_col = field_name.replace('_frequencies', '')
        _, freqs = compute_histogram_bins_and_freqs(df, base_col); return freqs
        
    if field_name.startswith('group_count_by_'): 
        group_col = field_name.replace('group_count_by_', '')
        _, values = compute_group_count(df, group_col)
        return values

    # Fallback for ungrouped aggregations by pattern
    if field_name.startswith('sum_'):
        col = field_name.replace('sum_', '')
        return compute_sum(df, col)
    if field_name.startswith('mean_'):
        col = field_name.replace('mean_', '')
        return compute_mean(df, col)
    if field_name.startswith('median_'):
        col = field_name.replace('median_', '')
        return compute_median(df, col)

    if field_name.startswith('correlation_'):
        parts = field_name.replace('correlation_', '').split('_')
        if len(parts) >= 2: 
            # Attempt to reconstruct column names if they contain underscores
            # This is a best-effort for fallback; explicit derived_from is preferred
            for i in range(1, len(parts)):
                col1_candidate = "_".join(parts[:i])
                col2_candidate = "_".join(parts[i:])
                if col1_candidate in df.columns and col2_candidate in df.columns:
                    return compute_correlation(df, col1_candidate, col2_candidate)
    return None 