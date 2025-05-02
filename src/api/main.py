from fastapi import FastAPI, UploadFile, File, HTTPException
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

        # Create the prompt
        print("📝 Creating prompt...")
        prompt = f"""Based on the provided data, generate 5 visualizations in JSON format. Each visualization should be appropriate for the data types and include actual data values.

Data Sample (first 20 rows):
{df.head(20).to_string()}

Data Analysis:
- Numeric columns: {', '.join(numeric_columns)}
- Categorical columns: {', '.join(categorical_columns)}
- Row count: {analysis['row_count']}

Create these specific visualizations:

1. For numeric columns: Create a histogram/frequency distribution showing the distribution of values
2. For categorical columns: Create a bar chart showing counts by category
3. For numeric columns: Create a box plot showing the distribution statistics
4. If there are both numeric and categorical columns: Create a grouped bar chart
5. For categorical data: Create a pie chart showing proportions (limit to top 5-7 categories if there are too many)

Each visualization specification must follow this exact structure:

{{
    "title": "string",
    "description": "string",
    "type": "bar|line|scatter|pie|box",
    "data": {{
        "x": [...],  // Actual data values, not column names
        "y": [...],  // Actual data values, not column names
        "labels": [...],  // For pie charts: actual category names
        "values": [...]  // For pie charts: actual numeric values
    }},
    "style": {{
        "color_scheme": "string",
        "show_values": boolean,
        "show_legend": boolean
    }},
    "layout": {{
        "title": "string",
        "xaxis_title": "string",
        "yaxis_title": "string"
    }}
}}

For histograms/frequency distributions:
1. Calculate appropriate bin ranges for numeric data
2. Count the frequency of values in each bin
3. Use these as x (bin ranges) and y (frequencies) values

For categorical data:
1. Count the frequency of each category
2. Use actual category names (not A, B, C, etc.)
3. Sort by frequency for better visualization

IMPORTANT:
1. Return ONLY a JSON array with 5 visualizations
2. Include actual data values in x, y, labels, and values fields
3. Ensure all numeric values are numbers, not strings
4. Do not include any explanation text, only the JSON array
5. Make sure the JSON is properly formatted and valid
6. Use only existing columns from the data
7. For pie charts, use actual category names, not generic labels like 'Category A'
8. For histograms, calculate and provide actual bin ranges and frequencies"""
        print("✅ Prompt created")

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
            print("📝 Raw response content:", content[:500] + "..." if len(content) > 500 else content)
            
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
                
                if not isinstance(visualizations, list):
                    raise ValueError("Response is not a JSON array")
                
                # Validate and clean each visualization
                cleaned_visualizations = []
                for idx, viz in enumerate(visualizations):
                    print(f"🔍 Processing visualization {idx + 1}")
                    if not isinstance(viz, dict):
                        print(f"❌ Skipping invalid visualization {idx + 1}: {viz}")
                        continue
                    # Ensure required fields exist
                    required_fields = ['title', 'type', 'data']
                    if not all(field in viz for field in required_fields):
                        print(f"❌ Skipping visualization {idx + 1} missing required fields: {viz}")
                        continue
                    # Ensure data field has required arrays
                    if not isinstance(viz['data'], dict):
                        print(f"❌ Skipping visualization {idx + 1} with invalid data field: {viz}")
                        continue
                    # Convert numeric strings to numbers
                    if 'data' in viz:
                        for key in list(viz['data'].keys()):
                            if isinstance(viz['data'][key], list):
                                viz['data'][key] = [
                                    float(val) if is_number(val) else val
                                    for val in viz['data'][key]
                                ]
                    # Thorough fix for box plot: ensure y is present and valid, remove extra fields
                    if viz.get('type') == 'box':
                        if 'y' not in viz['data'] or not isinstance(viz['data']['y'], list) or not viz['data']['y']:
                            print(f"❌ Skipping box plot visualization {idx + 1} due to missing or empty y array.")
                            continue
                        # Remove any extra fields like quartiles
                        keys_to_remove = [k for k in viz['data'] if k not in ['y']]
                        for k in keys_to_remove:
                            viz['data'].pop(k)
                    # Thorough fix for bar chart: always return a data array of traces
                    if viz.get('type') == 'bar':
                        traces = []
                        # Grouped bar: look for y1, y2, ...
                        y_keys = [k for k in viz['data'].keys() if k.startswith('y') and k != 'y']
                        if y_keys:
                            # Use x from data
                            x = viz['data'].get('x', [])
                            for y_key in sorted(y_keys):
                                y = viz['data'][y_key]
                                name = viz['data'].get('names', {}).get(y_key) or y_key
                                traces.append({
                                    'type': 'bar',
                                    'name': name,
                                    'x': x,
                                    'y': y
                                })
                            # Remove legacy y1, y2, ... fields
                            for y_key in y_keys:
                                viz['data'].pop(y_key)
                            if 'x' in viz['data']:
                                viz['data'].pop('x')
                            if 'names' in viz['data']:
                                viz['data'].pop('names')
                        elif 'x' in viz['data'] and 'y' in viz['data']:
                            # Single bar chart
                            traces.append({
                                'type': 'bar',
                                'x': viz['data']['x'],
                                'y': viz['data']['y'],
                                'name': viz.get('title', 'Bar')
                            })
                            viz['data'].pop('x')
                            viz['data'].pop('y')
                        viz['data'] = traces
                    print(f"✅ Successfully processed visualization {idx + 1}")
                    cleaned_visualizations.append(viz)
                
                if not cleaned_visualizations:
                    raise ValueError("No valid visualizations found in response")
                
                print(f"✅ Final processed visualizations: {json.dumps(cleaned_visualizations, indent=2)}")
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
        # Read the file content
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(pd.io.common.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        analysis = analyze_data(df)
        visualizations = generate_insights(df)
        return JSONResponse(content={
            "status": "success",
            "analysis": analysis,
            "visualizations": visualizations
        })
    except Exception as e:
        print(f"❌ Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 