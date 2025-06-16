import streamlit as st
import pandas as pd
import json
from langchain.tools import Tool
from langchain.chains import LLMChain
from utils.code_utils import extract_code, extract_explanation
from prompts.templates import viz_prompt
from tools.viz_tool import create_viz_tool

def create_google_charts_tool(llm):
    """Creates a visualization tool using Google Charts API with Plotly fallback."""
    viz_chain = LLMChain(llm=llm, prompt=viz_prompt)
    plotly_tool = create_viz_tool(llm)  # Create Plotly tool for fallback

    def get_google_charts_code(chart_type, data, options):
        """Generate Google Charts JavaScript code."""
        data_json = data.to_json(orient='records')
        
        chart_html = f"""
        <div id="chart_div" style="width: 100%; height: 500px;"></div>
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
            google.charts.load('current', {{'packages':['corechart']}});
            google.charts.setOnLoadCallback(drawChart);

            function drawChart() {{
                var data = new google.visualization.DataTable();
                
                // Add columns
                {json.dumps([{"label": col, "type": "number" if pd.api.types.is_numeric_dtype(data[col]) else "string"} 
                            for col in data.columns])}
                data.addRows({data_json});

                var options = {json.dumps(options)};

                var chart = new google.visualization.{chart_type}(document.getElementById('chart_div'));
                chart.draw(data, options);
            }}
        </script>
        """
        return chart_html

    def viz_tool_func(request: str) -> str:
        df = st.session_state.data.copy()
        cols = ", ".join(df.columns.tolist())

        # Generate LLM response
        llm_response = viz_chain.run(columns=cols, request=request)
        st.subheader("[Viz Tool] LLM Raw Response")
        st.markdown(llm_response)

        try:
            # Extract visualization parameters from LLM response
            code_snippet = extract_code(llm_response)
            if not code_snippet:
                return "Error: No visualization code found in agent's response"

            # Execute the code to get visualization parameters
            namespace = {
                'pd': pd, 'df': df, 'st': st,
                'chart_type': None,
                'selected_columns': None,
                'options': None
            }
            
            exec(code_snippet, {}, namespace)
            
            chart_type = namespace.get('chart_type')
            selected_columns = namespace.get('selected_columns')
            options = namespace.get('options', {})
            
            if not all([chart_type, selected_columns]):
                return "Error: Agent did not specify chart type or columns"
            
            # Prepare data for visualization
            viz_data = df[selected_columns].copy()
            
            try:
                # Try Google Charts
                chart_html = get_google_charts_code(chart_type, viz_data, options)
                st.components.v1.html(chart_html, height=550)
                return f"Created {chart_type} using Google Charts API"
                
            except Exception as google_error:
                st.warning(f"Google Charts failed: {str(google_error)}. Falling back to Plotly...")
                # Use the existing Plotly tool as fallback
                return plotly_tool.func(request)

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return f"Error: {str(e)}"

    return Tool(
        name="Google Charts Visualization Tool",
        func=viz_tool_func,
        description="""Create interactive visualizations using Google Charts API with Plotly fallback.
        The agent will determine the appropriate chart type, columns, and options based on the data and request.
        Input: visualization request string (e.g., 'Show a line chart of sales over time')"""
    ) 