from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import anthropic
import json
from datetime import datetime
import tempfile
import plotly.graph_objects as go
import shutil
import scipy.stats
from itertools import combinations
import re
from derived_fields import resolve_derived_field
from chart_processing import process_visualizations
import redis
import uuid
import logging

load_dotenv()

app = FastAPI(title="ChartSage API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002"],  # Add your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Claude client
claude = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Initialize Redis client (local instance)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

_last_uploaded_df = None

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chartsage.log', mode='a', encoding='utf-8')
    ]
)

# Redirect print to logging.info
import builtins
print = lambda *args, **kwargs: logging.info(' '.join(str(a) for a in args))

def serialize_pandas_data(df: pd.DataFrame) -> dict:
    """Convert pandas DataFrame to JSON-serializable format."""
    preview_data = []
    for _, row in df.head(10).iterrows():
        row_dict = {}
        for column in df.columns:
            value = row[column]
            # Handle timestamp/datetime objects
            if pd.isna(value):
                row_dict[column] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                row_dict[column] = value.isoformat()
            else:
                row_dict[column] = value
        preview_data.append(row_dict)
    
    return {
        "columns": df.columns.tolist(),
        "data": preview_data
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"📁 Starting file upload process for: {file.filename}")
        
        # Read the file content
        content = await file.read()
        print("✅ File content read successfully")
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(pd.io.common.BytesIO(content), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(pd.io.common.BytesIO(content), encoding='cp1252')
                except Exception:
                    df = pd.read_csv(pd.io.common.BytesIO(content), encoding='latin1')
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(pd.io.common.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        print(f"✅ DataFrame loaded with shape: {df.shape}")
        
        # Lowercase all DataFrame column names immediately after loading any file (CSV or Excel)
        df.columns = [col.lower() for col in df.columns]
        
        # Store the DataFrame globally
        global _last_uploaded_df
        _last_uploaded_df = df
        
        # Get data preview using the serializer
        preview_data = serialize_pandas_data(df)
        print("✅ Data preview generated")
        
        # Return just the preview data initially
        response_data = {
            "status": "success",
            "preview": preview_data
        }
        print("✅ Response prepared, sending to client...")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Error in upload process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """Endpoint to download generated PDF files."""
    file_path = f"temp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="PDF file not found")

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the data structure and return metadata."""
    analysis = {
        "columns": [],
        "row_count": len(df),
        "data_types": {},
        "missing_values": {},
        "basic_stats": {}
    }
    
    for column in df.columns:
        col_type = str(df[column].dtype)
        missing = df[column].isnull().sum()
        stats = {}
        
        if np.issubdtype(df[column].dtype, np.number):
            stats = {
                "mean": float(df[column].mean()),
                "median": float(df[column].median()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max())
            }
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            stats = {
                "min_date": str(df[column].min()),
                "max_date": str(df[column].max())
            }
        else:
            stats = {
                "unique_values": int(df[column].nunique()),
                "most_common": {str(k): v for k, v in df[column].value_counts().head(3).to_dict().items()}
            }
        
        analysis["columns"].append(column)
        analysis["data_types"][column] = col_type
        analysis["missing_values"][column] = int(missing)
        analysis["basic_stats"][column] = stats
    
    return analysis

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

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

def compute_correlation(df, col1, col2):
    """Compute correlation between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return []
    return df[col1].corr(df[col2])

def process_derived_field(field_name, df):
    """Process any derived field based on naming patterns."""
    # Handle ratio fields (e.g., 'column1_to_column2_ratio')
    if '_to_' in field_name and '_ratio' in field_name:
        parts = field_name.replace('_ratio', '').split('_to_')
        if len(parts) == 2 and all(p in df.columns for p in parts):
            return compute_ratio(df, parts[0], parts[1])
    
    # Handle histogram fields (e.g., 'column_bins' or 'column_frequencies')
    if field_name.endswith('_bins') or field_name.endswith('_frequencies'):
        base_col = field_name.replace('_bins', '').replace('_frequencies', '')
        if base_col in df.columns:
            bins, freqs = compute_histogram_bins_and_freqs(df, base_col)
            return bins if field_name.endswith('_bins') else freqs
    
    # Handle group count fields (e.g., 'group_count_by_column')
    if field_name.startswith('group_count_by_'):
        group_col = field_name.replace('group_count_by_', '')
        if group_col in df.columns:
            labels, values = compute_group_count(df, group_col)
            return labels, values
    
    # Handle budget categories
    if field_name == 'budget_categories':
        categories, _ = compute_budget_categories(df)
        return categories
    if field_name == 'budget_category_counts':
        _, counts = compute_budget_categories(df)
        return counts
    
    # Handle performance segments
    if field_name == 'performance_segment':
        segments, _ = compute_performance_segments(df)
        return segments
    if field_name == 'segment_count':
        _, counts = compute_performance_segments(df)
        return counts
    
    # If it's a direct column reference
    if field_name in df.columns:
        return df[field_name].tolist()
    
    return None

def validate_and_fix_field(field, df, field_key=None):
    # If it's a string, return as is (will be handled downstream)
    if isinstance(field, str):
        return field
    # If it's a list of single characters, treat as malformed
    if isinstance(field, list) and all(isinstance(x, str) and len(x) == 1 for x in field):
        print(f"[WARN] Field '{field_key}' is a list of single characters, likely malformed. Skipping.")
        return None
    # If it's a list, only allow for certain derived/summary fields
    if isinstance(field, list):
        if field_key and is_allowed_array_field(field_key):
            return field
        print(f"[WARN] Field '{field_key}' is a list but not allowed for this field. Skipping.")
        return None
    # Otherwise, skip
    print(f"[WARN] Field '{field_key}' is not a recognized type. Skipping.")
    return None

def get_field_as_list(field, df, valid_indices=None, field_key=None, chart_obj=None):
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
        # If it's a list of numbers, use as-is
        if all(isinstance(f, (int, float, np.integer, np.floating)) for f in field):
            values = [float(f) if isinstance(f, np.floating) else int(f) if isinstance(f, np.integer) else f for f in field]
        # If it's a list of strings and this is the labels field, use as-is
        elif all(isinstance(f, str) for f in field) and field_key == "labels":
            values = field
        # If it's a list of strings for x/y/values, resolve each
        elif all(isinstance(f, str) for f in field):
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
                        if isinstance(derived, list) and len(derived) == 1:
                            resolved.append(derived[0])
                        elif isinstance(derived, list) and all(isinstance(x, (int, float, str, np.integer, np.floating)) for x in derived):
                            # If it's a list of values, flatten for bar/pie
                            resolved.extend([float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x for x in derived])
                        else:
                            resolved.append(derived)
                    else:
                        print(f"[WARN] Could not resolve field '{f}' in list for '{field_key}'. Skipping.")
                        resolved.append(None)
            values = resolved
        else:
            return []
    else:
        values = []
    if valid_indices is not None and isinstance(values, list):
        return [values[i] for i in valid_indices if i < len(values)]
    return values

def generate_insights(df: pd.DataFrame) -> List[Dict]:
    """Generate visualization specifications using Claude."""
    try:
        print("🔍 Starting generate_insights function")
        # Convert DataFrame to dict for JSON serialization
        df_dict = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df_dict[column] = df[column].tolist()
            else:
                df_dict[column] = df[column].astype(str).tolist()
        print("✅ Converted DataFrame to dict")
        # Create analysis for better visualization suggestions
        analysis = analyze_data(df)
        numeric_columns = [col for col, dtype in analysis["data_types"].items() 
                         if dtype.startswith(('int', 'float'))]
        categorical_columns = [col for col, dtype in analysis["data_types"].items() 
                            if not dtype.startswith(('int', 'float'))]
        print(f"✅ Found {len(numeric_columns)} numeric columns and {len(categorical_columns)} categorical columns")
        # Compute detailed summary statistics
        summary_lines = []
        # Basic stats
        for col in df.columns:
            if col in numeric_columns:
                summary_lines.append(f"- {col} (numeric): min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}, missing={df[col].isnull().sum()}")
            elif col in categorical_columns:
                top_cats = df[col].value_counts().head(5)
                top_cats_str = ', '.join([f'{k} ({v})' for k, v in top_cats.items()])
                summary_lines.append(f"- {col} (categorical): top categories: {top_cats_str}, missing={df[col].isnull().sum()}")
            else:
                summary_lines.append(f"- {col}: missing={df[col].isnull().sum()}")
        # Advanced numeric stats
        if numeric_columns:
            corr = df[numeric_columns].corr().round(2)
            summary_lines.append(f"Correlation matrix:\n{corr.to_string()}")
            for col in numeric_columns:
                skew = df[col].skew()
                kurt = df[col].kurt()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                outliers = ((df[col] < (df[col].quantile(0.25) - 1.5 * iqr)) | (df[col] > (df[col].quantile(0.75) + 1.5 * iqr))).sum()
                summary_lines.append(f"- {col}: skew={skew:.2f}, kurtosis={kurt:.2f}, IQR={iqr:.2f}, outliers={outliers}")
        # Advanced categorical stats
        for col in categorical_columns:
            cardinality = df[col].nunique()
            if len(df[col].value_counts(normalize=True)) > 0:
                top_cat_pct = df[col].value_counts(normalize=True).iloc[0] * 100
                entropy = scipy.stats.entropy(df[col].value_counts(normalize=True))
            else:
                top_cat_pct = 0
                entropy = 0
            summary_lines.append(f"- {col}: cardinality={cardinality}, top category %={top_cat_pct:.1f}, entropy={entropy:.2f}")
        # Group-wise stats
        for cat_col in categorical_columns:
            for num_col in numeric_columns:
                group_means = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(3)
                summary_lines.append(f"- {num_col} by {cat_col} (top 3 means): {group_means.to_dict()}")
        # Correlation between categorical columns (chi-squared)
        if len(categorical_columns) >= 2:
            for col1, col2 in combinations(categorical_columns, 2):
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p, _, _ = scipy.stats.chi2_contingency(contingency)
                    summary_lines.append(f"- {col1} vs {col2}: chi2={chi2:.2f}, p={p:.3f}")
        summary_stats = '\n'.join(summary_lines)
        # Load prompt template from file
        prompt_path = os.path.join(os.path.dirname(__file__), 'insight_prompt.txt')
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            data_sample=df.head(20).to_string(),
            numeric_columns=', '.join(numeric_columns),
            categorical_columns=', '.join(categorical_columns),
            row_count=analysis['row_count'],
            summary_stats=summary_stats
        )
        print("✅ Prompt loaded and filled")
        # Log the full prompt sent to Claude
        print(f"📝 PROMPT SENT TO CLAUDE (first 1000 chars):\n{prompt[:1000]}{'...' if len(prompt) > 1000 else ''}")
        print("🤖 Sending request to Claude API...")
        try:
            print("📊 Data shape:", df.shape)
            print("📊 Data columns:", df.columns.tolist())
            print("📊 Data types:", df.dtypes.to_dict())
            
            response = claude.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=4000,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print("✅ Received response from Claude API")
            
            # Extract and clean the response
            content = response.content[0].text.strip()
            print(f"📝 RAW CLAUDE RESPONSE (first 1000 chars):\n{content[:1000]}{'...' if len(content) > 1000 else ''}")
            # Ensure we have a JSON array
            if not content.startswith('['):
                content = content[content.find('['):]
            if not content.endswith(']'):
                content = content[:content.rfind(']')+1]
            print("📝 Cleaned response content:", content[:500] + "..." if len(content) > 500 else content)
            
            # Parse and validate the JSON
            try:
                visualizations = json.loads(content)
                print(f"✅ Successfully parsed JSON with {len(visualizations)} items")
                # Ensure all traces in all charts have their column names resolved to arrays
                for viz in visualizations:
                    if isinstance(viz.get('data'), list):
                        viz['data'] = [resolve_column_names(trace, df) for trace in viz['data']]
                    elif isinstance(viz.get('data'), dict):
                        viz['data'] = resolve_column_names(viz['data'], df)
                log_visualizations_list_concisely(visualizations, "✅ Visualizations after resolving column names:")
                if not isinstance(visualizations, list):
                    raise ValueError("Response is not a JSON array")
                
                # Process all visualizations using the new process_visualizations function
                cleaned_visualizations = process_visualizations(visualizations, df)
                
                if not cleaned_visualizations:
                    raise ValueError("No valid visualizations found in response")
                
                log_visualizations_list_concisely(cleaned_visualizations, "✅ Final processed visualizations:")
                # Clean non-JSON-compliant floats before returning
                cleaned_visualizations = clean_json_floats(cleaned_visualizations)
                # Add a helper to recursively convert all numpy types to native Python types for JSON serialization
                cleaned_visualizations = [make_json_serializable(viz) for viz in cleaned_visualizations]
                return cleaned_visualizations
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing Claude response: {str(e)}")
                print(f"Invalid JSON content: {content}")
                raise HTTPException(status_code=500, detail="Failed to parse visualization specifications")
                
        except Exception as api_error:
            # Check for overloaded error (529 or overloaded in message)
            if (hasattr(api_error, 'status_code') and api_error.status_code == 529) or \
               ("overloaded" in str(api_error).lower()) or ("529" in str(api_error)):
                print("Claude API overloaded, sending busy signal to frontend.")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "busy",
                        "message": "All our AI agents are busy at the moment. Please wait 30 seconds and we will try again automatically."
                    }
                )
            print(f"❌ Claude API Error: {str(api_error)}")
            print(f"❌ Error type: {type(api_error).__name__}")
            import traceback
            print(f"❌ API error traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Claude API Error: {str(api_error)}")

    except HTTPException as http_exc:
        # Propagate HTTPException (e.g., 503 busy) to FastAPI
        raise http_exc
    except Exception as e:
        print(f"❌ Error in generate_insights: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Generate insights traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.post("/generate-dashboard")
async def generate_dashboard(file: UploadFile = File(...)):
    """Generate dashboard visualizations from uploaded file (stateless)."""
    try:
        content = await file.read()
        import pandas as pd
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(pd.io.common.BytesIO(content))
        else:
            return JSONResponse(content={"status": "error", "message": "Unsupported file format"}, status_code=400)
        # Lowercase all DataFrame column names immediately after loading any file (CSV or Excel)
        df.columns = [col.lower() for col in df.columns]
        analysis = analyze_data(df)
        try:
            visualizations = generate_insights(df)
        except HTTPException as http_exc:
            # Propagate HTTPException (e.g., 503 busy) to FastAPI
            raise http_exc
        return JSONResponse(content={
            "status": "success",
            "analysis": analysis,
            "visualizations": visualizations
        })
    except HTTPException as http_exc:
        # Propagate HTTPException (e.g., 503 busy) to FastAPI
        raise http_exc
    except Exception as e:
        print(f"[ERROR] Failed to process /generate-dashboard POST: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# Add helper to clean non-JSON-compliant floats
def clean_json_floats(obj):
    if isinstance(obj, list):
        return [clean_json_floats(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_json_floats(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

# Helper to recursively resolve column names to arrays in all traces
def resolve_column_names(obj, df):
    """Resolve column names to arrays, handling derived fields."""
    if isinstance(obj, list):
        return [resolve_column_names(item, df) for item in obj]
    elif isinstance(obj, dict):
        # Handle data fields (x, y, labels, values)
        for key in ['x', 'y', 'labels', 'values']:
            if key in obj and isinstance(obj[key], str):
                colname = obj[key]
                print(f"[RESOLVING_FIELD] Chart field processing: {key}={colname}")
                
                # Real column
                if colname in df.columns:
                    obj[key] = df[colname].tolist()
                    continue
                
                # Histogram fields
                if colname.endswith('_bins') or colname.endswith('_frequencies'):
                    base_col = colname.replace('_bins', '').replace('_frequencies', '')
                    if base_col in df.columns:
                        bins, freqs = compute_histogram_bins_and_freqs(df, base_col)
                        obj[key] = bins if colname.endswith('_bins') else freqs
                        continue
                
                # Ratio fields
                if '_to_' in colname and '_ratio' in colname:
                    parts = colname.replace('_ratio', '').split('_to_')
                    if len(parts) == 2 and all(p in df.columns for p in parts):
                        obj[key] = compute_ratio(df, parts[0], parts[1])
                        continue
                
                # Group count fields
                if colname.startswith('group_count_by_'):
                    group_col = colname.replace('group_count_by_', '')
                    if group_col in df.columns:
                        labels, values = compute_group_count(df, group_col)
                        obj[key] = labels if key in ['x', 'labels'] else values
                        continue
                
                print(f"[WARN] Could not resolve field: {colname}")
        
        # Recursively process nested objects
        for k, v in obj.items():
            obj[k] = resolve_column_names(v, df)
        return obj
    return obj

def compute_histogram_bins_and_freqs(df, col, bins=10):
    """Compute histogram bins and frequencies, robustly handling data types."""
    print(f"[HISTOGRAM] Processing column: {col} with dtype: {df[col].dtype}")
    series = df[col].dropna()
    
    if len(series) == 0:
        print(f"[HISTOGRAM] Column {col} is empty after dropna.")
        return [], []

    numeric_values_for_hist = None
    is_datetime_type = False

    # Attempt to convert to datetime
    # Make a copy to avoid SettingWithCopyWarning if series is a view
    series_copy = series.copy() 
    series_dt_coerced = pd.to_datetime(series_copy, errors='coerce')

    if pd.api.types.is_datetime64_any_dtype(series_dt_coerced.dtype) and not series_dt_coerced.isnull().all():
        # Successfully converted to datetime and not all values are NaT
        valid_datetimes = series_dt_coerced.dropna()
        if len(valid_datetimes) > 0:
            numeric_values_for_hist = valid_datetimes.astype(np.int64) // 10**9  # Convert ns to s
            is_datetime_type = True
            print(f"[HISTOGRAM] Column {col} treated as DATETIME.")
        else:
             print(f"[HISTOGRAM] Column {col} converted to datetime but all values became NaT or NaN.")
    
    if numeric_values_for_hist is None: # Did not become valid datetime
        # Check if original (or coerced if not datetime) is numeric
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric_values_for_hist = series.astype(float) # Ensure float for histogram
            print(f"[HISTOGRAM] Column {col} treated as NUMERIC (original dtype).")
        elif pd.api.types.is_numeric_dtype(series_dt_coerced.dtype): # e.g. if to_datetime produced numbers
             numeric_values_for_hist = series_dt_coerced.dropna().astype(float)
             print(f"[HISTOGRAM] Column {col} treated as NUMERIC (after to_datetime coerce gave numeric).")
        else:
            # If still not numeric (e.g. object type with mixed non-convertible strings)
            print(f"[HISTOGRAM] Column '{col}' (dtype: {series.dtype}) is not convertible to datetime or numeric. Cannot generate histogram.")
            return [], []

    if numeric_values_for_hist is None or len(numeric_values_for_hist) == 0 : 
        print(f"[HISTOGRAM] Column {col} resulted in no valid numeric values for histogram.")
        return [], []
        
    # Use numpy's histogram with automatic bin selection
    try:
        counts, bin_edges = np.histogram(numeric_values_for_hist, bins='auto')
    except TypeError as e:
        print(f"[ERROR] np.histogram failed for column '{col}'. Processed values head: {numeric_values_for_hist.head() if hasattr(numeric_values_for_hist, 'head') else str(numeric_values_for_hist)[:100]}. Error: {e}")
        return [], []
    
    # Format bin labels
    if is_datetime_type:
        if len(bin_edges) < 2:
            print(f"[HISTOGRAM] Not enough bin edges for column {col} to form labels (datetime).")
            return [], []
        bin_labels = [f'{pd.to_datetime(bin_edges[i], unit="s").strftime("%Y-%m-%d")} to {pd.to_datetime(bin_edges[i+1], unit="s").strftime("%Y-%m-%d")}' 
                      for i in range(len(bin_edges)-1)]
    else: # Numeric type
        if len(bin_edges) < 2:
            print(f"[HISTOGRAM] Not enough bin edges for column {col} to form labels (numeric).")
            return [], []
        if ('amount' in col.lower() or 'price' in col.lower() or 'currency' in col.lower() or '$' in col.lower()) and \
           all(isinstance(x, (int, float)) for x in bin_edges) and \
           all(x >= 0 for x in bin_edges):
             bin_labels = [f'${int(bin_edges[i]):,}-${int(bin_edges[i+1]):,}' for i in range(len(bin_edges)-1)]
        else: 
             bin_labels = [f'{float(bin_edges[i]):,.2f}-{float(bin_edges[i+1]):,.2f}' for i in range(len(bin_edges)-1)]

    return bin_labels, counts.tolist()

def compute_ratio(df, num_col, denom_col):
    """Compute ratio between two columns, handling edge cases."""
    if num_col not in df.columns or denom_col not in df.columns:
        return []
    # Handle division by zero and NaN values
    ratio = df[num_col] / df[denom_col].replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()

def compute_group_count(df, group_col):
    counts = df[group_col].value_counts()
    return counts.index.tolist(), counts.values.tolist()

def compute_budget_categories(df):
    """Compute budget categories and their counts."""
    if 'budget' not in df.columns:
        return [], []
    
    # Define budget ranges
    ranges = [
        (0, 50_000_000, "Under $50M"),
        (50_000_000, 100_000_000, "$50M-$100M"),
        (100_000_000, 150_000_000, "$100M-$150M"),
        (150_000_000, 200_000_000, "$150M-$200M"),
        (200_000_000, float('inf'), "Over $200M")
    ]
    
    # Count movies in each range
    categories = []
    counts = []
    for start, end, label in ranges:
        count = len(df[(df['budget'] >= start) & (df['budget'] < end)])
        if count > 0:  # Only include non-empty categories
            categories.append(label)
            counts.append(count)
    
    return categories, counts

def compute_performance_segments(df):
    """Compute performance segments based on box office revenue."""
    if 'box_office_worldwide' not in df.columns:
        return [], []
    
    # Define performance segments
    ranges = [
        (0, 100_000_000, "Under $100M"),
        (100_000_000, 300_000_000, "$100M-$300M"),
        (300_000_000, 500_000_000, "$300M-$500M"),
        (500_000_000, 1_000_000_000, "$500M-$1B"),
        (1_000_000_000, float('inf'), "Over $1B")
    ]
    
    # Count movies in each segment
    segments = []
    counts = []
    for start, end, label in ranges:
        count = len(df[(df['box_office_worldwide'] >= start) & (df['box_office_worldwide'] < end)])
        if count > 0:  # Only include non-empty segments
            segments.append(label)
            counts.append(count)
    
    return segments, counts

@app.post("/visualizations")
async def visualizations(request: Request):
    """Accepts a JSON body with a 'data' field containing visualization data."""
    try:
        body = await request.json()
        data = body.get('data')
        if not data:
            return JSONResponse(content={"status": "error", "message": "No data provided."}, status_code=400)
        # Here you would process the data as needed, e.g., validate, transform, or store
        # For now, just echo back the data for testing
        return JSONResponse(content={"status": "success", "visualizations": data})
    except Exception as e:
        print(f"[ERROR] Failed to process /visualizations POST: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# --- VALIDATION HELPERS ---
ALLOWED_ARRAY_FIELDS = {'bins', 'frequencies', 'group_counts', 'top_n', 'labels', 'values'}

def is_allowed_array_field(field_name):
    # Accept arrays only for bins, frequencies, group counts, top-N summaries, labels, or values
    for allowed in ALLOWED_ARRAY_FIELDS:
        if allowed in field_name:
            return True
    return False

# Add a helper to recursively convert all numpy types to native Python types for JSON serialization
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

@app.post("/session-dashboard")
async def save_session_dashboard(request: Request):
    data = await request.json()
    session_id = str(uuid.uuid4())
    redis_client.set(f"dashboard:{session_id}", json.dumps(data), ex=60*60*24)  # 24 hour expiry
    return {"session_id": session_id}

@app.get("/session-dashboard/{session_id}")
async def get_session_dashboard(session_id: str):
    raw = redis_client.get(f"dashboard:{session_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return json.loads(raw)

def log_chart_json(chart_json):
    # Helper to log only the first 10 items of x, y, labels, values
    def truncate(arr):
        if isinstance(arr, list):
            return arr[:10] + ([f"...+{len(arr)-10} more"] if len(arr) > 10 else [])
        return arr
    if isinstance(chart_json, dict):
        for key in ['x', 'y', 'labels', 'values']:
            if key in chart_json:
                chart_json[key] = truncate(chart_json[key])
        print(json.dumps(chart_json, indent=2, default=str))
    else:
        print(chart_json)

def log_visualizations_list_concisely(viz_list, context_message=""):
    if not viz_list:
        print(f"{context_message} Empty list.")
        return

    print(f"{context_message} (Showing summary of {len(viz_list)} visualizations):")
    summaries = []
    for i, viz_item in enumerate(viz_list):
        if isinstance(viz_item, dict):
            summary_item = {}
            for k, v in viz_item.items():
                if k in ['x', 'y', 'labels', 'values', 'text'] and isinstance(v, list): # Add 'text' here
                    summary_item[k] = v[:5] + ([f"...+{len(v)-5} more"] if len(v) > 5 else [])
                elif isinstance(v, str) and len(v) > 100: # Truncate long strings like 'explanation' or title
                    summary_item[k] = v[:100] + "..."
                elif isinstance(v, dict) and k == 'data': # Handle nested 'data' dict
                    nested_summary = {}
                    for nk, nv in v.items():
                        if nk in ['x', 'y', 'labels', 'values', 'text'] and isinstance(nv, list):
                             nested_summary[nk] = nv[:5] + ([f"...+{len(nv)-5} more"] if len(nv) > 5 else [])
                        elif isinstance(nv, str) and len(nv) > 100:
                            nested_summary[nk] = nv[:100] + "..."
                        else:
                            nested_summary[nk] = nv
                    summary_item[k] = nested_summary
                else:
                    summary_item[k] = v
            summaries.append(summary_item)
        else: 
            # Should not happen if AI response and processing is correct, but good to handle
            try:
                summaries.append(str(viz_item)[:200] + "..." if len(str(viz_item)) > 200 else str(viz_item))
            except Exception:
                summaries.append("Unstringable item in visualization list")
    try:
        # Attempt to pretty print the summarized list
        print(json.dumps(summaries, indent=2, default=str))
    except TypeError: 
        # Fallback if json.dumps fails (e.g. before make_json_serializable)
        # This fallback might be less readable but better than crashing the log
        print("Summary (fallback string representation):")
        for s_item in summaries:
            print(str(s_item))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 