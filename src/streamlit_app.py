import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anthropic import Anthropic
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import numpy as np

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="ChartSage - Interactive Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
                "most_common": df[column].value_counts().head(3).to_dict()
            }
        
        analysis["columns"].append(column)
        analysis["data_types"][column] = col_type
        analysis["missing_values"][column] = int(missing)
        analysis["basic_stats"][column] = stats
    
    return analysis

def get_visualization_suggestions(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get visualization suggestions from Claude."""
    prompt = f"""Analyze this dataset and suggest 5 different visualizations based on the data patterns. For each visualization, provide a detailed specification in JSON format.

Available visualization types:
1. Line Chart - Best for temporal trends and continuous data over time
2. Bar Chart - Best for comparing categories or showing distribution
3. Scatter Plot - Best for showing relationships between two numeric variables
4. Pie Chart - Best for showing composition when there are few categories
5. Box Plot - Best for showing distribution and outliers in numeric data

Data Summary:
{json.dumps(analysis, indent=2)}

Return ONLY a JSON array of 5 visualization suggestions, each with these exact fields:
{{
    "title": "string (clear, concise title for the visualization)",
    "type": "string (one of: line, bar, scatter, pie, box)",
    "description": "string (explain what insights this visualization reveals)",
    "data_requirements": {{
        "x_axis": "string (column name for x-axis or categories)",
        "y_axis": "string (column name for y-axis or values)",
        "group_by": "string (optional column name for grouping/series)",
        "aggregation": "string (e.g., sum, average, count)"
    }},
    "style_preferences": {{
        "color_scheme": "string (suggested color palette theme)",
        "orientation": "string (vertical or horizontal)",
        "show_legend": boolean,
        "include_gridlines": boolean,
        "show_data_labels": boolean
    }},
    "confidence": number (0-1)
}}

Ensure each visualization type is used appropriately for the data types available and provides meaningful insights.
"""
    
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        return json.loads(response.content[0].text)
    except Exception as e:
        st.error(f"Error parsing Claude's response: {str(e)}")
        return []

def create_visualization(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    """Create a plotly figure based on the visualization specification."""
    viz_type = spec['type']
    data_req = spec['data_requirements']
    style = spec['style_preferences']
    
    # Apply aggregation if specified
    if data_req.get('aggregation'):
        df = aggregate_data(df, data_req)
    
    # Create the visualization based on type
    if viz_type == 'line':
        fig = px.line(
            df,
            x=data_req['x_axis'],
            y=data_req['y_axis'],
            color=data_req.get('group_by'),
            title=spec['title']
        )
    elif viz_type == 'bar':
        fig = px.bar(
            df,
            x=data_req['x_axis'],
            y=data_req['y_axis'],
            color=data_req.get('group_by'),
            title=spec['title']
        )
    elif viz_type == 'scatter':
        fig = px.scatter(
            df,
            x=data_req['x_axis'],
            y=data_req['y_axis'],
            color=data_req.get('group_by'),
            title=spec['title']
        )
    elif viz_type == 'pie':
        fig = px.pie(
            df,
            values=data_req['y_axis'],
            names=data_req['x_axis'],
            title=spec['title']
        )
    elif viz_type == 'box':
        fig = px.box(
            df,
            x=data_req.get('group_by'),
            y=data_req['y_axis'],
            title=spec['title']
        )
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
    
    # Apply styling
    fig.update_layout(
        template='plotly_white',
        showlegend=style['show_legend'],
        xgrid=style['include_gridlines'],
        ygrid=style['include_gridlines']
    )
    
    return fig

def aggregate_data(df: pd.DataFrame, data_req: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate data based on specified requirements."""
    agg_func = data_req['aggregation'].lower()
    group_cols = [data_req['x_axis']]
    
    if data_req.get('group_by'):
        group_cols.append(data_req['group_by'])
    
    if agg_func == 'sum':
        return df.groupby(group_cols)[data_req['y_axis']].sum().reset_index()
    elif agg_func == 'average':
        return df.groupby(group_cols)[data_req['y_axis']].mean().reset_index()
    elif agg_func == 'count':
        return df.groupby(group_cols)[data_req['y_axis']].count().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

# Main app layout
st.title("📊 ChartSage")
st.write("Upload your data file and let AI help you create meaningful visualizations.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Analyze data
        with st.spinner("Analyzing data..."):
            analysis = analyze_data(df)
        
        # Get visualization suggestions
        with st.spinner("Getting visualization suggestions..."):
            viz_specs = get_visualization_suggestions(df, analysis)
        
        # Create tabs for different visualizations
        if viz_specs:
            tabs = st.tabs([spec['title'] for spec in viz_specs])
            
            for tab, spec in zip(tabs, viz_specs):
                with tab:
                    # Show visualization description
                    st.write(spec['description'])
                    
                    # Create and display the visualization
                    try:
                        fig = create_visualization(df, spec)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show customization options
                        with st.expander("Customize Visualization"):
                            st.json(spec)
                            if st.button("Regenerate", key=f"regenerate_{spec['title']}"):
                                fig = create_visualization(df, spec)
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}") 