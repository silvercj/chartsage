import logging
import numpy as np

def detect_field_type(field):
    """
    Detects the type of a field for chart data.
    Returns one of: 'column_name', 'derived_field', 'column_names', 'derived_fields', 'bin_labels', 'values', 'frequencies', 'raw_values', 'unknown'.
    Logs the detected type.
    """
    # Helper: is it a derived field name?
    def is_derived(s):
        if not isinstance(s, str):
            return False
        patterns = [
            '_to_' in s and s.endswith('_ratio'),
            s.endswith('_bins'),
            s.endswith('_frequencies'),
            s.startswith('group_count_by_'),
            s.startswith('sum_'),
            s.startswith('mean_'),
            s.startswith('median_'),
            s.startswith('correlation_'),
        ]
        return any(patterns)

    # Single string
    if isinstance(field, str):
        if is_derived(field):
            logging.info(f"[FIELD TYPE] Detected single derived field: {field}")
            return 'derived_field'
        else:
            logging.info(f"[FIELD TYPE] Detected single column name: {field}")
            return 'column_name'
    # List/array
    elif isinstance(field, list):
        if all(isinstance(f, str) for f in field):
            if all(is_derived(f) for f in field):
                logging.info(f"[FIELD TYPE] Detected array of derived fields: {field}")
                return 'derived_fields'
            # Heuristic: bin labels often contain $ and - or are numeric-like strings
            is_potential_bin_label_list = True
            for f_item in field:
                f_cleaned = f_item.replace(',', '').replace('-', '').replace('$', '').replace(' ', '')
                # Check for currency range, simple number string, or date-like range "YYYY-MM-DD to YYYY-MM-DD"
                # or simple numeric range like "10-20"
                if not ( ('$' in f_item and '-' in f_item) or 
                         f_cleaned.isdigit() or 
                         (f_item.count(' to ') == 1 and len(f_item.split(' to ')[0]) >= 8 and len(f_item.split(' to ')[1]) >= 8) or # Date range heuristic
                         (f_item.count('-') == 1 and all(part.strip().replace(',','').isdigit() for part in f_item.split('-') if part.strip())) ): # Numeric range "10-20", "1,000-2,000"
                    is_potential_bin_label_list = False
                    break
            if is_potential_bin_label_list and len(field) > 0: # ensure field is not empty for this heuristic
                logging.info(f"[FIELD TYPE] Detected bin labels: {field[:5]}...")
                return 'bin_labels'
            
            # If it's not derived and not bin labels, but a list of strings,
            # treat as raw string values. Avoid assuming 'column_names' too readily.
            logging.info(f"[FIELD TYPE] Detected array of strings (treating as raw_values): {field[:5]}...")
            return 'raw_values'

        elif all(isinstance(f, (int, float, np.integer, np.floating)) for f in field):
            logging.info(f"[FIELD TYPE] Detected array of raw values (numeric): {field[:5]}...")
            return 'raw_values'
        else:
            logging.info(f"[FIELD TYPE] Detected array of mixed/unknown type (attempting raw_values): {field[:5]}...")
            # Attempt to convert all to string and treat as raw_values if items are simple
            try:
                str_list = [str(s) for s in field]
                # Check if all items were successfully converted to string (basic check)
                # This path is for lists of mixed types or types not explicitly handled above.
                # We assume if they can be stringified, they might be usable as raw categorical labels.
                logging.info(f"[FIELD TYPE] Mixed list, elements stringified, treating as raw_values: {[str(s)[:50] for s in str_list][:5]}...")
                return 'raw_values' 
            except Exception as e:
                logging.warning(f"[FIELD TYPE] Failed to convert mixed list to strings for raw_values: {e}. Falling back to unknown.")
                pass # If conversion fails, fall through to unknown
            logging.info(f"[FIELD TYPE] Detected array of unknown type (fallback): {field[:5]}...")
            return 'unknown'
    else:
        # For single items or non-list/non-string types not caught above
        type_name = type(field).__name__
        logging.info(f"[FIELD TYPE] Detected unknown field structure (type: {type_name}, value: {str(field)[:100]}{'...' if len(str(field)) > 100 else ''})")
        return 'unknown' 