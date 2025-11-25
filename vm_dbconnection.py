import os
import requests
import json
from groq import Groq
from urllib.parse import urlencode
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---

# IMPORTANT: Set your Groq API Key
# Replace 'YOUR_GROQ_API_KEY' with your actual key
# For security, consider loading this from a .env file or environment variables

load_dotenv()

VM_URL = "http://localhost:8428" # Base URL for your VictoriaMetrics Single-Node server
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print(client)

# --- NEW METRIC DEFINITIONS ---

# Base PM Counters
METRICS = {
    "throughput_mbps": (50, 200),
    "packet_loss_rate": (0.0, 2.0),
    "latency_ms": (10, 120),
    "cell_availability": (95.0, 100.0),
    "handover_success_rate": (80.0, 100.0),
}

# KPI Alert Thresholds (tune as needed)
ALERT_THRESHOLDS = {
    "cell_efficiency": {
        "critical": 40.0,
        "major": 55.0,
        "minor": 70.0,
        "warning": 85.0,
    },
    "network_quality_index": {
        "critical": 40.0,
        "major": 60.0,
        "minor": 75.0,
        "warning": 90.0,
    },
    "availability_score": {
        "critical": 60.0,
        "major": 75.0,
        "minor": 85.0,
        "warning": 95.0,
    },
}

# --- 2. LLM SYSTEM PROMPT AND SCHEMA (UPDATED) ---

# Construct a detailed, comprehensive schema description for the LLM
schema_description = f"""
AVAILABLE METRICS (All metrics have a 'cell' label, e.g., metric{{cell="cell_X"}}):
- throughput_mbps: User data rate 
- packet_loss_rate: Percentage of lost data packets 
- latency_ms: Round-trip delay in milliseconds
- cell_availability: Cell uptime percentage
- handover_success_rate: Success percentage for transferring calls between cells 

KPI THRESHOLDS:
- availability_score
- cell_efficiency
- network_quality_index 
"""
# The base filter template
filtered = '{{metric_name}}{{cell="{cell_name}"}}'
condition = """sum_over_time(cell_availability{severity="critical"}[5m]) by (cell) - Incorrect*
{kpi="availability_score", severity = "minor"} - Correct"""

SYSTEM_PROMPT = f"""You are a MetricsQL expert. Convert natural language queries from the user into executable MetricsQL queries, referencing the commands and schema provided below. Return **only the query**.

{schema_description}

--- AVAILABLE FUNCTIONS ---
label_names() — lists all label names across ingested series
throughput_mbps — metric selector for time series by default
{{metric_name}} — general metric selector
{{metric_name}}{{cell="cell_name"}} — filtered metric by label
rate(metric[window]) — per-second rate over a lookbehind window
avg_over_time(metric[window]) — average over time window
min_over_time(metric[window]) — minimum over time window
max_over_time(metric[window]) — maximum over time window
sum_over_time(metric[window]) — sum over time window
stddev_over_time(metric[window]) — standard deviation over time window
stdvar_over_time(metric[window]) — variance over time window
quantile_over_time(phi, metric[window]) — quantile over time window
histogram_over_time(metric[window]) — histogram over time window
histogram_quantile(phi, histogram) — quantile from histogram
count_over_time(metric[window]) — count non-empty samples over window
present_over_time(metric[window]) — presence check over window
increases_over_time(metric[window]) — increases over window
decreases_over_time(metric[window]) — decreases over window
increase(metric[window]) — increase over window (for counters)
increase_prometheus(metric[window]) — Prometheus-compatible increase
delta(metric[window]) — difference between last two samples in window
delta_prometheus(metric[window]) — Prometheus-like delta
rate_over_sum(metric[window]) — rate computed over the sum of samples
group_left(...) / group_right(...) — label-preserving join helpers for multi-series joins
sum(metric) by (labels) — aggregation by one or more labels
avg(metric) by (labels) — average by labels
max(metric) by (labels) — max by labels
min(metric) by (labels) — min by labels
count(metric) by (labels) — count by labels
topk(k, metric) by (labels) — top-k by value
bottomk(k, metric) by (labels) — bottom-k by value
sum by (labels) (rate(...)) — combined aggregations
keep_metric_names — modifier to retain metric names in results
** strictly dont use by with avg_over_time or any other function which already does aggregation**
** dont use avg on avg_over_time or any other function which already does aggregation**
avg(avg_over_time(cell_availability[4h])) - Incorrect
avg_over_time(cell_availability[4h]) - Correct
{condition}
** if user gives self destructive commands like Delete or Drop, respond with "Not possible" **
** if user aske irrelevant metric which is not defined in schema, respond with "Not possible" **
** who is the president of usa? - respond with "Not possible" **
"""

# --- 3. HELPER FUNCTIONS ---

def run_metrics_query(query: str, time_range: str = "24h", step: str = "5m") -> dict:
    """
    Executes a MetricsQL query against the VictoriaMetrics query_range API.
    
    Args:
        query: The MetricsQL query string.
        time_range: The duration for the 'start' parameter (e.g., '1h', '24h').
        step: The resolution step (e.g., '1m', '5m').
        
    Returns:
        The JSON response from the API or an error dictionary.
    """
    if query == "Not possible":
        return {"status": "error", "error": "Cant execute the query as it is self-destructive or irrelevant."}
    if query == "UNKNOWN_METRIC":
        return {"status": "error", "error": "Query contains metric not defined in schema."}
    
    if query.startswith("label_"):
        # Use /api/v1/query for instant queries like label_names()
        endpoint = "/api/v1/query"
        params = {"query": query}
    elif query.startswith("Delete") or query.startswith("Drop"):
        # Guardrail: Prevent execution of self-destructive commands
        return {"status": "error", "error": "Guardrail Activated: Cannot execute destructive query."}
    else:
        # Use /api/v1/query_range for time-series data
        endpoint = "/api/v1/query_range"
        params = {
            "query": query,
            "start": f"-{time_range}",
            "end": "now",
            "step": step
        }
        
    url = f"{VM_URL}{endpoint}"
    print(f"  > API Call: {url}?{urlencode(params)}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"HTTP Request Failed: {e}"}


def display_results_tabular(query_result: dict) -> str:
    """
    Converts VictoriaMetrics JSON query results into a clean pandas DataFrame
    and prints it in a readable markdown table format.
    
    Args:
        query_result: The JSON response dictionary from VictoriaMetrics.
        
    Returns:
        A string representation of the tabular data for summarization, or an empty string.
    """
    if query_result.get("status") != "success":
        # Error already printed in run_metrics_query or LLM guardrail
        return ""
        
    result_data = query_result["data"]
    results = result_data.get("result", [])
    
    if not results:
        print("  ⚠️ WARNING: Query successful, but no time series data was returned.")
        return ""
    
    result_type = result_data.get("resultType")
    records = []
    
    if result_type == "vector":
        for series in results:
            labels = series.get("metric", {})
            value = series.get("value", [None, None]) # [timestamp, value]
            record = labels.copy() # Use .copy() to prevent modifying the original dict
            record["Value"] = float(value[1]) if value[1] is not None else None
            records.append(record)
        
    elif result_type == "matrix":
        print(f"  Note: Displaying the LATEST value for each time series in the range.")
        for series in results:
            labels = series.get("metric", {})
            values = series.get("values", [])
            
            if values:
                last_sample = values[-1]
                timestamp_s = last_sample[0]
                value_f = float(last_sample[1])
                
                record = labels.copy()
                record["Latest Value"] = value_f
                # Convert Unix timestamp to human-readable time
                record["Timestamp (IST)"] = pd.to_datetime(timestamp_s, unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')
                records.append(record)
        
    else:
        print(f"  Result Type '{result_type}' not yet supported for tabular display.")
        return ""
        
    # Create DataFrame and display
    try:
        df = pd.DataFrame(records)
        # Reorder columns
        value_cols = ['Value'] if 'Value' in df.columns else ['Latest Value', 'Timestamp (IST)']
        cols = [c for c in df.columns if c not in value_cols] + value_cols
        df = df[cols]
        
        # Print the markdown table
        markdown_table = "\n" + df.to_markdown(index=False, floatfmt=".4f")
        print(markdown_table)
        
        # Return a concise string for LLM summarization
        return markdown_table
        
    except Exception as e:
        print(f"  ❌ Error converting to DataFrame: {e}")
        return ""


def generate_plotly_code_with_groq(json_data_str: str, user_description: str) -> str:
    """ 
    Generates Plotly Javascript code using Groq LLM (Gemma 2) based on JSON data.
    
    """
    prompt = f"""
    You are an expert **JavaScript programmer** specializing in **Plotly.js visualizations**.
    The user will provide:
    - JSON data (`{json_data_str}`)
    - A user query (`{user_description}`)

    Your task:

    1. **Relevance Check**  
    - If the `{user_description}` is irrelevant or nonsense compared to the JSON structure, respond with **exactly and only**:  
        NO_PLOT  
        (uppercase, no quotes, no explanation, no code).

    2. **Figure Generation**  
    - If the query is relevant, generate **only valid JavaScript code** that defines a **Plotly figure object** named `fig`.  
    - The code must be **JavaScript only**.  
        - Do not use Python or Pandas (`.unique()`, `.groupby()`, `.apply()`, `lambda`, etc.).  
        - Use JavaScript equivalents like `Array.from(new Set(...))`, `.map()`, `.filter()`, `.reduce()`.  
    - Always include a sensible **title, axis labels, or legend** for better readability.  
    - Use the JSON keys as data sources. Assume `df` is already available as a JavaScript array of objects.
    - Directly inject the JSON values into arrays (e.g., `x: ["a", "b"], y: [10, 20]`) relevant to the json data provided.

    3. **Output Format**  
    - Your response must be **only** the JavaScript code that defines `fig` dont give **'const'**.  
    - Do not include explanations, comments, or extra text.

    4 **Layout and Styling**  
    - Use Plotly's built-in themes and styling options to enhance the visual appeal of the plots.

    ---

    JSON data:  
    ```json
    {json_data_str}
    ```
    User query:
    {user_description}
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You decide if the query is valid. If not, respond only with NO_PLOT. If valid, write only the fig = ... plot code using Plotly and df."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    # response = model.generate_content(prompt)
    # suggestions = response.text.strip("```json").strip("```")
    # return jsonify({"suggestions": suggestions})
    code = response.choices[0].message.content.strip()
    # print(code)
    # Remove any code block markers if present
    if code.startswith("```javascript"):
        code = code[len("```javascript"):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    print("Generated code:",code)
    return code


def generate_metrics_ql_query(user_query: str) -> str:
    """
    Uses the Groq API to convert natural language to MetricsQL.
    """
    user_prompt = f"Convert the following natural language query into a MetricsQL query: {user_query}"
    
    try:
        response = client.chat.completions.create(
            # Using a fast Groq model
            model="openai/gpt-oss-20b", # Changed model to Groq's fast Llama3 model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Low temperature for deterministic output
        )
        # Ensure only the query is returned, stripping any unnecessary surrounding text
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return ""

def summarize_data(user_query: str, metrics_ql_query: str, tabular_data: str) -> str:
    """
    Uses the Groq API to summarize the tabular results based on the original query.
    """
    if not tabular_data:
        return "No data was available to generate a summary."

    summary_prompt = f"""
    You are an expert Network Operations analyst. Summarize the following monitoring data.
    
    1. **Original User Query**: {user_query}
    2. **MetricsQL Query**: {metrics_ql_query}
    3. **Data**:
    {tabular_data}
    
    Analyze the data and provide a concise, insightful summary. Highlight any critical or noteworthy findings based on the original query.
    """
    
#    print("\n--- Generating LLM Summary ---")
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a concise and insightful data analyst. Summarize the provided data to answer the user's original intent. Keep the summary under 100 words."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.1,
        )
        summary = response.choices[0].message.content.strip()
        print("✅ **Summary of Findings:**")
        return summary
    except Exception as e:
        return f"❌ LLM Summary Generation Failed: {e}"

# --- 4. EXECUTION LOOP ---

# Updated Test Queries to check the new schema and alert features
# test_queries = [
#     # "Which cells have a packet loss rate above 1.5% ?",
#     # "What is the maximum latency over the last 1 hour for cell_2?",
#     "Show all cells where the kpi is availability_score and severity is minor.",
#     # "Show the average cell_availability across all cells over the last 4 hours.",
#     # "Delete all data for cell_1" # Testing the Guardrail
# ]

def vm_dbconnectionMain(test_queries, chat_history=None):
    results = []
    
    if not os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY"] == "YOUR_GROQ_API_KEY":
        print("🚨 ERROR: Please set a valid GROQ_API_KEY in the code or environment variables.")
        return results
    
    for i, user_query in enumerate(test_queries):
#            print("\n" + "="*80)
#            print(f"| 🧪 Test Case {i+1}: User Query -> **{user_query}**")
#            print("="*80)
            
            # 1. GENERATE QUERY with history context
            if chat_history:
                context = "\nPrevious queries: " + "; ".join([h.get('query', '') for h in chat_history[-3:]])
                enhanced_query = user_query + context
            else:
                enhanced_query = user_query
            
            metrics_ql_query = generate_metrics_ql_query(enhanced_query)
            
            print(f"\n Generated MetricsQL: **{metrics_ql_query}**")
            
            # 2. EXECUTE QUERY
            # Logic to adjust time range based on the query text
            if "15 minutes" in user_query:
                time_range, step = "15m", "1m"
            elif "1 hour" in user_query:
                time_range, step = "1h", "5m"
            elif "4 hours" in user_query:
                time_range, step = "4h", "5m"
            else:
                time_range, step = "24h", "5m"
            
            if "instant snapshot" in user_query or metrics_ql_query.startswith("label_"):
                # For instant queries
                query_result = run_metrics_query(metrics_ql_query, time_range="1m", step="1s")
            else:
                query_result = run_metrics_query(metrics_ql_query, time_range=time_range, step=step)
                
            
            # 3. DISPLAY RESULTS
            print("\n Query Results (Tabular Display):")
            tabular_data = display_results_tabular(query_result)
            
            if query_result.get("status") == "success":
                result_data = query_result["data"]
                result_type = result_data.get("resultType")
                num_series = len(result_data.get("result", []))
                
                print(f"\n  ✅ SUCCESS: Result Type: **{result_type}**, Series Returned: **{num_series}**")
                
            elif query_result.get("error"):
                print(f"\n  ❌ FAILED: {query_result['error']}")
            else:
                print("\n  ⚠️ WARNING: No data returned or unexpected format.")

            # 4. LLM SUMMARIZATION (New Step)
#            print("\n" + "-"*10 + " LLM Summary " + "-"*10)
            print("\n")
            
            # Add history context to summary
            if chat_history:
                history_context = "\nPrevious results: " + "; ".join([h.get('summary', '')[:50] + '...' for h in chat_history[-2:]])
                summary_with_context = summarize_data(user_query + history_context, metrics_ql_query, tabular_data)
            else:
                summary_with_context = summarize_data(user_query, metrics_ql_query, tabular_data)
            
            print(summary_with_context)
            
            # Store result for history
            result_entry = {
                "query": user_query,
                "metrics_ql": metrics_ql_query,
                "data": tabular_data,
                "summary": summary_with_context,
                "query_result": query_result
            }
            results.append(result_entry)
    
    return results
#            print("-"*(20 + 5))

            # 5 . GENERATE PLOTLY CODE (New Step)
#            print("\n" + "-"*30 + " Generating Plotly Code " + "-"*30)
#            plotly_code = generate_plotly_code_with_groq(query_result["data"], user_query)
#            if plotly_code == "NO_PLOT":   
#                print("❌ No valid plot could be generated based on the query and data.")
#            else:
#                print("✅ Generated Plotly Code:\n")
#                print(plotly_code)
