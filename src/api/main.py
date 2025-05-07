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

_last_uploaded_df = None

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

def get_field_as_list(field, df, valid_indices=None, field_key=None):
    # If it's a string and matches a column, return the column as a list
    if isinstance(field, str):
        if field in df.columns:
            values = df[field].tolist()
        else:
            # For derived fields (e.g., bins/frequencies), just return as-is
            return field
    elif isinstance(field, list):
        if field_key and is_allowed_array_field(field_key):
            values = field
        else:
            # Not allowed, return empty
            return []
    else:
        values = []
    if valid_indices is not None and isinstance(values, list):
        return [values[i] for i in valid_indices if i < len(values)]
    return values

def process_visualization(viz, df, idx=None):
    chart_type = viz.get('type')
    title = viz.get('title', f'Chart {idx+1}' if idx is not None else 'Unknown')
    print(f"\n🔍 Processing visualization {idx+1 if idx is not None else ''}: {title} (type: {chart_type})")
    if chart_type == 'bar':
        return process_bar_chart(viz, df, idx)
    elif chart_type == 'scatter':
        return process_scatter_chart(viz, df, idx)
    elif chart_type == 'pie':
        return process_pie_chart(viz, df, idx)
    elif chart_type == 'box':
        return process_box_chart(viz, df, idx)
    else:
        print(f"[WARN] Unknown chart type: {chart_type} (chart {idx+1 if idx is not None else ''})")
        return None

def process_bar_chart(viz, df, idx=None):
    data = viz['data']
    # Remove 'type' from data if present
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Bar {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    # Accept arrays for bins/frequencies, else use column
    x_vals = get_field_as_list(data.get('x'), df)
    y_vals = get_field_as_list(data.get('y'), df)
    if isinstance(x_vals, str) or isinstance(y_vals, str):
        print(f"[ERROR] (Bar {idx+1}) x or y is a derived field name or missing. Skipping.")
        return None
    if not isinstance(x_vals, list) or not isinstance(y_vals, list) or not x_vals or not y_vals:
        print(f"[ERROR] (Bar {idx+1}) x or y is not a valid list. Skipping.")
        return None
    data['x'] = x_vals
    data['y'] = y_vals
    print(f"[INFO] (Bar {idx+1}) x: {x_vals[:5]}... y: {y_vals[:5]}...")
    viz['data'] = data
    return viz

def process_scatter_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Scatter {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    x_vals = get_field_as_list(data.get('x'), df)
    y_vals = get_field_as_list(data.get('y'), df)
    if isinstance(x_vals, str) or isinstance(y_vals, str):
        print(f"[ERROR] (Scatter {idx+1}) x or y is a derived field name or missing. Skipping.")
        return None
    valid_indices = [i for i, (x, y) in enumerate(zip(x_vals, y_vals)) if not (pd.isna(x) or pd.isna(y))]
    if not valid_indices:
        print(f"[ERROR] (Scatter {idx+1}) No valid data points after NaN filtering. Skipping.")
        return None
    data['x'] = [x_vals[i] for i in valid_indices]
    data['y'] = [y_vals[i] for i in valid_indices]
    if 'labels' in data:
        data['labels'] = get_field_as_list(data['labels'], df, valid_indices)
    if 'text' in data and isinstance(data['text'], list):
        data['text'] = [data['text'][i] for i in valid_indices if i < len(data['text'])]
    else:
        if 'labels' in data:
            data['text'] = [
                f"{label}<br>X: {format_currency(x)}<br>Y: {format_currency(y)}"
                for label, x, y in zip(data['labels'], data['x'], data['y'])
            ]
    print(f"[INFO] (Scatter {idx+1}) x: {data['x'][:5]}... y: {data['y'][:5]}... labels: {data.get('labels', [])[:5]}")
    viz['data'] = data
    return viz

def process_pie_chart(viz, df, idx=None):
    data = viz['data']
    if isinstance(data, dict) and 'type' in data:
        print(f"[INFO] (Pie {idx+1}) Removing redundant 'type' field from data object.")
        del data['type']
    labels = get_field_as_list(data.get('labels'), df)
    values = get_field_as_list(data.get('values'), df)
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
    x_vals = get_field_as_list(data.get('x'), df)
    y_vals = get_field_as_list(data.get('y'), df)
    if isinstance(x_vals, str) or isinstance(y_vals, str):
        print(f"[ERROR] (Box {idx+1}) x or y is a derived field name or missing. Skipping.")
        return None
    if not isinstance(x_vals, list) or not isinstance(y_vals, list) or not x_vals or not y_vals:
        print(f"[ERROR] (Box {idx+1}) x or y is not a valid list. Skipping.")
        return None
    data['x'] = x_vals
    data['y'] = y_vals
    print(f"[INFO] (Box {idx+1}) x: {x_vals[:5]}... y: {y_vals[:5]}...")
    viz['data'] = data
    return viz

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
        print("📝 FULL PROMPT SENT TO CLAUDE:\n" + prompt)
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
            print("📝 RAW CLAUDE RESPONSE (FULL):\n" + content)
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
                print("✅ After resolving column names:", json.dumps(visualizations, indent=2))
                if not isinstance(visualizations, list):
                    raise ValueError("Response is not a JSON array")
                # Validate and clean each visualization
                cleaned_visualizations = []
                for idx, viz in enumerate(visualizations):
                    print(f"🔍 Processing visualization {idx + 1}")
                    processed_viz = process_visualization(viz, df, idx)
                    if processed_viz:
                        cleaned_visualizations.append(processed_viz)
                
                if not cleaned_visualizations:
                    raise ValueError("No valid visualizations found in response")
                
                print(f"✅ Final processed visualizations: {json.dumps(cleaned_visualizations, indent=2)}")
                # Clean non-JSON-compliant floats before returning
                cleaned_visualizations = clean_json_floats(cleaned_visualizations)
                return cleaned_visualizations
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing Claude response: {str(e)}")
                print(f"Invalid JSON content: {content}")
                raise HTTPException(status_code=500, detail="Failed to parse visualization specifications")
                
        except Exception as api_error:
            print(f"❌ Claude API Error: {str(api_error)}")
            print(f"❌ Error type: {type(api_error).__name__}")
            import traceback
            print(f"❌ API error traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Claude API Error: {str(api_error)}")

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
        analysis = analyze_data(df)
        visualizations = generate_insights(df)
        return JSONResponse(content={
            "status": "success",
            "analysis": analysis,
            "visualizations": visualizations
        })
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
                print(f"[DERIVED] Processing field: {key}={colname}")
                
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
    """Compute histogram bins and frequencies, handling NaN values."""
    values = df[col].dropna()
    if len(values) == 0:
        return [], []
    # Use numpy's histogram with automatic bin selection
    counts, bin_edges = np.histogram(values, bins='auto')
    # Format bin labels with proper number formatting
    bin_labels = [f'${int(bin_edges[i]):,}-${int(bin_edges[i+1]):,}' for i in range(len(bin_edges)-1)]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 