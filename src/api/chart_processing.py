import numpy as np
import pandas as pd
from derived_fields import resolve_derived_field
from field_type_utils import detect_field_type

def format_currency(value):
    """Format number as currency string."""
    if pd.isna(value):
        return "N/A"
    try:
        if isinstance(value, str):
            # Try to convert string to float first
            value = float(value)
        return f"${int(value):,}"
    except (ValueError, TypeError):
        return str(value)

def format_ratio(value):
    """Format number as ratio with 2 decimal places."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}x"

def format_short_currency(value):
    """Format number as $1.7B, $200M, etc."""
    if pd.isna(value):
        return "N/A"
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"${int(value):,}"
        else:
            return f"${int(value)}"
    except Exception:
        return str(value)

def get_field_as_list(field, df, valid_indices=None, field_key=None, chart_obj=None):
    # Check for type hint in chart_obj
    type_hint = None
    if chart_obj and field_key:
        type_hint = chart_obj.get(f"{field_key}_type")
    if isinstance(field, str):
        if field in df.columns:
            values = df[field].tolist()
        else:
            derived = resolve_derived_field(field, df, chart_obj=chart_obj)
            if derived is not None:
                print(f"[INFO] Resolved derived field '{field}' for '{field_key}'.")
                values = derived if isinstance(derived, list) else [derived]
            else:
                print(f"[WARN] Could not resolve field '{field}' for '{field_key}'. Skipping.")
                return []
    elif isinstance(field, list):
        # If a type hint is present, use it to decide how to handle the array
        if type_hint:
            if type_hint in ('bin_labels', 'values', 'frequencies', 'derived'):
                values = field
            elif type_hint == 'column_names':
                # Try to resolve each as a column
                resolved = []
                for f in field:
                    if f in df.columns:
                        col_vals = df[f].tolist()
                        print(f"[INFO] Resolved column '{f}' for '{field_key}'.")
                        resolved.append(col_vals)
                    else:
                        derived = resolve_derived_field(f, df, chart_obj=chart_obj)
                        if derived is not None:
                            print(f"[INFO] Resolved derived field '{f}' in list for '{field_key}'.")
                            resolved.append(derived)
                        else:
                            print(f"[WARN] Could not resolve field '{f}' in list for '{field_key}'. Skipping.")
                            resolved.append(None)
                values = resolved
            else:
                print(f"[WARN] Unrecognized type hint '{type_hint}' for field '{field_key}'. Using array as-is.")
                values = field
        else:
            # Use detect_field_type to determine how to handle the array
            detected_type = detect_field_type(field)
            print(f"[INFO] Detected field type for '{field_key}': {detected_type}")
            if detected_type in ('bin_labels', 'values', 'frequencies', 'derived', 'raw_values'):
                values = field
            elif detected_type == 'column_names':
                resolved = []
                for f in field:
                    if f in df.columns:
                        col_vals = df[f].tolist()
                        print(f"[INFO] Resolved column '{f}' for '{field_key}'.")
                        resolved.append(col_vals)
                    else:
                        derived = resolve_derived_field(f, df, chart_obj=chart_obj)
                        if derived is not None:
                            print(f"[INFO] Resolved derived field '{f}' in list for '{field_key}'.")
                            resolved.append(derived)
                        else:
                            print(f"[WARN] Could not resolve field '{f}' in list for '{field_key}'. Skipping.")
                            resolved.append(None)
                values = resolved
            elif detected_type == 'derived_fields':
                resolved = []
                for f in field:
                    derived = resolve_derived_field(f, df, chart_obj=chart_obj)
                    if derived is not None:
                        print(f"[INFO] Resolved derived field '{f}' in list for '{field_key}'.")
                        resolved.append(derived)
                    else:
                        print(f"[WARN] Could not resolve derived field '{f}' in list for '{field_key}'. Skipping.")
                        resolved.append(None)
                values = resolved
            else:
                print(f"[WARN] Unhandled detected field type '{detected_type}' for field '{field_key}'. Using array as-is.")
                values = field
    else:
        values = []
    if valid_indices is not None and isinstance(values, list):
        return [values[i] for i in valid_indices if i < len(values)]
    return values

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def process_scatter_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Scatter {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    x_vals = get_field_as_list(data.get('x'), df, field_key='x', chart_obj=viz)
    y_vals = get_field_as_list(data.get('y'), df, field_key='y', chart_obj=viz)
    if isinstance(x_vals, str) or isinstance(y_vals, str):
        print(f"[ERROR] (Scatter {idx+1}) x or y is a derived field name or missing. Skipping.")
        return None
    valid_indices = [i for i, (x, y) in enumerate(zip(x_vals, y_vals)) if not (pd.isna(x) or pd.isna(y))]
    if not valid_indices:
        print(f"[ERROR] (Scatter {idx+1}) No valid data points after NaN filtering. Skipping.")
        return None
    data['x'] = [x_vals[i] for i in valid_indices]
    data['y'] = [y_vals[i] for i in valid_indices]
    labels = get_field_as_list(data.get('labels'), df, valid_indices, field_key='labels', chart_obj=viz) if 'labels' in data else None
    n_points = len(data['x'])
    if n_points > 10:
        y_array = np.array(data['y'])
        top5_idx = y_array.argsort()[-5:][::-1]
        text = ['' for _ in range(n_points)]
        label_list = ['' for _ in range(n_points)]
        for i in top5_idx:
            label = labels[i] if labels else ''
            text[i] = f"{label}<br>Budget: {format_short_currency(data['x'][i])}<br>Box Office: {format_short_currency(data['y'][i])}" if label else f"Budget: {format_short_currency(data['x'][i])}<br>Box Office: {format_short_currency(data['y'][i])}"
            label_list[i] = label
        data['text'] = text
        data['labels'] = label_list
    else:
        if labels:
            data['text'] = [
                f"{label}<br>Budget: {format_short_currency(x)}<br>Box Office: {format_short_currency(y)}" if label else f"Budget: {format_short_currency(x)}<br>Box Office: {format_short_currency(y)}"
                for label, x, y in zip(labels, data['x'], data['y'])
            ]
        else:
            data['text'] = [f"Budget: {format_short_currency(x)}<br>Box Office: {format_short_currency(y)}" for x, y in zip(data['x'], data['y'])]
        data['labels'] = labels if labels else None
    print(f"[INFO] (Scatter {idx+1}) x: {data['x'][:5]}... y: {data['y'][:5]}... labels: {data['labels'][:5]}")
    viz['data'] = data
    return viz

def process_bar_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Bar {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    x_vals = get_field_as_list(data.get('x'), df, field_key='x', chart_obj=viz)
    y_vals = get_field_as_list(data.get('y'), df, field_key='y', chart_obj=viz)
    if isinstance(x_vals, str) or isinstance(y_vals, str):
        print(f"[ERROR] (Bar {idx+1}) x or y is a derived field name or missing. Skipping.")
        return None
    if not isinstance(x_vals, list) or not isinstance(y_vals, list) or not x_vals or not y_vals:
        print(f"[ERROR] (Bar {idx+1}) x or y is not a valid list. Skipping.")
        return None
    data['x'] = x_vals
    data['y'] = y_vals
    if 'labels' in data:
        data['labels'] = get_field_as_list(data['labels'], df, field_key='labels', chart_obj=viz)
    print(f"[INFO] (Bar {idx+1}) x: {x_vals[:5]}... y: {y_vals[:5]}...")
    viz['data'] = data
    return viz

def process_pie_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Pie {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    labels = get_field_as_list(data.get('labels'), df, field_key='labels', chart_obj=viz)
    values = get_field_as_list(data.get('values'), df, field_key='values', chart_obj=viz)
    if isinstance(labels, str) or isinstance(values, str):
        print(f"[ERROR] (Pie {idx+1}) labels or values is a derived field name or missing. Skipping.")
        return None
    if not isinstance(labels, list) or not isinstance(values, list) or not labels or not values:
        print(f"[ERROR] (Pie {idx+1}) labels or values is not a valid list. Skipping.")
        return None
    data['labels'] = labels
    data['values'] = values
    print(f"[INFO] (Pie {idx+1}) labels: {labels[:5]}... values: {values[:5]}...")
    viz['data'] = data
    return viz

def process_box_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Box {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    x_vals = get_field_as_list(data.get('x'), df, field_key='x', chart_obj=viz)
    y_vals = get_field_as_list(data.get('y'), df, field_key='y', chart_obj=viz)
    print(f"[DEBUG] (Box {idx+1}) x: {x_vals}, y: {y_vals}")  # Extra logging
    # Accept box plot with only x or only y as a valid list
    if (isinstance(x_vals, list) and x_vals and (y_vals is None or not y_vals)):
        data['x'] = x_vals
        data['y'] = None
        print(f"[INFO] (Box {idx+1}) Using only x for box plot.")
    elif (isinstance(y_vals, list) and y_vals and (x_vals is None or not x_vals)):
        data['x'] = None
        data['y'] = y_vals
        print(f"[INFO] (Box {idx+1}) Using only y for box plot.")
    elif (isinstance(x_vals, list) and x_vals and isinstance(y_vals, list) and y_vals):
        data['x'] = x_vals
        data['y'] = y_vals
        print(f"[INFO] (Box {idx+1}) Using both x and y for box plot.")
    else:
        print(f"[ERROR] (Box {idx+1}) x and y are not valid lists. Skipping.")
        return None
    if 'labels' in data:
        data['labels'] = get_field_as_list(data['labels'], df, field_key='labels', chart_obj=viz)
    print(f"[INFO] (Box {idx+1}) x: {x_vals[:5]}... y: {y_vals[:5]}...")
    viz['data'] = data
    return viz

def process_visualizations(visualizations, df):
    processed = []
    for idx, viz in enumerate(visualizations):
        if not isinstance(viz, dict) or 'type' not in viz or 'data' not in viz:
            print(f"[WARN] Skipping invalid visualization object: {viz}")
            continue
        chart_type = viz['type']
        if chart_type == 'scatter':
            processed_viz = process_scatter_chart(viz, df, idx)
        elif chart_type == 'bar':
            processed_viz = process_bar_chart(viz, df, idx)
        elif chart_type == 'pie':
            processed_viz = process_pie_chart(viz, df, idx)
        elif chart_type == 'box':
            processed_viz = process_box_chart(viz, df, idx)
        else:
            print(f"[WARN] Unknown chart type: {chart_type} (chart {idx+1})")
            continue
        if processed_viz:
            processed_viz = make_json_serializable(processed_viz)
            processed.append(processed_viz)
    return processed 