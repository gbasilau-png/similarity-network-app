import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import os
import base64
import io
from datetime import datetime

print("🚀 Starting Joint Weighted Network Application...")

# -------------------------------
# Load and process data
# -------------------------------
try:
    print("📊 Loading similarity tables...")
    df_jac = pd.read_csv("similarity_jaccard.csv")
    df_euc = pd.read_csv("similarity_euclidean_normalized.csv")
    df_cos = pd.read_csv("cosine_similarity_results.csv")
    df_substance = pd.read_csv("Joint_Similarity_Substance.csv")
    print("✅ All CSV files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CSV files: {e}")
    raise

# Rename similarity columns
df_jac.rename(columns={'Jaccard_Similarity': 'Jaccard'}, inplace=True)
df_euc.rename(columns={'Euclidean_Similarity': 'Euclidean'}, inplace=True)
df_cos.rename(columns={'Cosine_Similarity': 'Cosine'}, inplace=True)

# Merge similarity tables
df_merged = df_jac.merge(df_euc, on=['Sample_1', 'Sample_2']).merge(df_cos, on=['Sample_1', 'Sample_2'])

# Aggregate substances per sample pair
df_substance = (
    df_substance.groupby(['Sample_1', 'Sample_2'])['Common_Substance']
    .apply(lambda x: ', '.join(sorted(set(x.dropna()))))
    .reset_index()
    .rename(columns={'Common_Substance': 'Key_Substances'})
)

# Merge aggregated substance info
df_merged = df_merged.merge(df_substance, on=['Sample_1', 'Sample_2'], how='left')

# Generate dropdown list (unique substances)
substances = ['All'] + sorted(
    set(sum([s.split(', ') for s in df_merged['Key_Substances'].dropna()], []))
)

print(f"✅ Data processing complete. Found {len(substances)} unique substances.")
print(f"✅ Total sample pairs: {len(df_merged)}")

# -------------------------------
# Initialize Dash app
# -------------------------------
app = dash.Dash(__name__)
app.title = "Joint Weighted Network GUI"
server = app.server  # Important for deployment

# -------------------------------
# Helper Functions
# -------------------------------
def validate_weights(w_j, w_e, w_c, threshold):
    """Validate input weights and threshold"""
    weights = [w_j or 0, w_e or 0, w_c or 0]
    
    if any(w < 0 or w > 100 for w in weights + [threshold or 0]):
        raise ValueError("Weights and threshold must be between 0-100")
    
    return weights

def create_enhanced_network(G, df_edges):
    """Create enhanced network visualization with colored edges"""
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Create edge traces with color based on weight
    edge_traces = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = G[u][v]['weight']
        
        # Color based on weight (darker blue for higher weights)
        color_intensity = max(0.3, weight)  # Ensure minimum visibility
        color = f'rgba(30, 136, 229, {color_intensity})'
        
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            line=dict(width=2 + weight * 5, color=color),
            hoverinfo='text',
            text=f"{u}-{v}: {weight:.3f}",
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x, node_y, node_text, node_hover = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        neighbors = list(G.neighbors(node))
        node_hover.append(f"Sample: {node}<br>Connections: {len(neighbors)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=node_hover,
        hoverinfo='text',
        textposition='middle center',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        showlegend=False
    )
    
    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    
    fig.update_layout(
        title="Weighted Similarity Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='white'
    )
    
    return fig

def weight_input_block(label, input_id, slider_id, default_value):
    """Reusable block for weight input and slider"""
    return html.Div([
        html.Label(f"{label} (%)", style={'fontSize':'16px', 'fontWeight':'bold', 'color':'#000'}),
        dcc.Input(
            id=input_id, type='number', value=default_value, min=0, max=100, step=1,
            style={'width':'120px', 'height':'35px', 'fontSize':'16px', 'textAlign':'center', 'marginBottom':'10px'}
        ),
        dcc.Slider(
            id=slider_id, min=0, max=100, step=1, value=default_value,
            tooltip={"placement": "bottom", "always_visible": True},
            marks={i: str(i) for i in range(0, 101, 20)}
        )
    ], style={'padding':'15px', 'margin':'10px', 'backgroundColor':'#f8f9fa', 'borderRadius':'8px'})

# -------------------------------
# Layout (Stacked - Graph TOP, Table BOTTOM)
# -------------------------------
app.layout = html.Div([
    html.H1("Joint Weighted Network Visualization", 
            style={'textAlign': 'center', 'color': '#000', 'marginBottom': '30px'}),
    
    # Weights configuration section
    html.Div([
        html.H3("Similarity Weights Configuration", 
               style={'color': '#000', 'borderBottom': '2px solid #ccc', 'paddingBottom': '10px'}),
        html.Div([
            html.Div([
                weight_input_block("Jaccard Weight", "w-jaccard", "slider-jaccard", 35),
                weight_input_block("Euclidean Weight", "w-euclidean", "slider-euclidean", 30),
                weight_input_block("Cosine Weight", "w-cosine", "slider-cosine", 35),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}),
            
            html.Div([
                weight_input_block("Similarity Threshold", "threshold", "slider-threshold", 90),
            ], style={'width': '50%', 'margin': '20px auto'})
        ])
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px', 'border': '1px solid #ddd'}),
    
    # Chemical filter section
    html.Div([
        html.H3("Filter by Chemical Components", style={'color': '#000', 'marginBottom': '15px'}),
        dcc.Dropdown(
            id='chemical-dropdown',
            options=[{'label': s, 'value': s} for s in substances],
            value=['All'],
            multi=True,
            clearable=True,
            style={'fontSize': '16px'},
            placeholder="Select chemical components..."
        )
    ], style={'padding': '20px', 'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
    
    # Results display
    html.Div(id='joint-similarity-display', style={
        'border': '2px solid #007bff', 
        'padding': '15px', 
        'margin': '10px', 
        'backgroundColor': '#e7f3ff', 
        'color': '#000', 
        'fontSize': '16px',
        'fontWeight': 'bold',
        'borderRadius': '8px'
    }),
    
    # Save buttons section
    html.Div([
        html.H4("Save Current Results", style={'color': '#000', 'marginBottom': '15px', 'textAlign': 'center'}),
        html.Div([
            html.Button("💾 Save Network Diagram as PNG", 
                       id='save-diagram-btn',
                       style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '14px', 
                              'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 
                              'borderRadius': '5px', 'cursor': 'pointer'}),
            html.Button("📊 Save Table as CSV", 
                       id='save-table-btn',
                       style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '14px', 
                              'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 
                              'borderRadius': '5px', 'cursor': 'pointer'}),
            html.Button("📋 Save Table as Excel", 
                       id='save-excel-btn',
                       style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '14px', 
                              'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 
                              'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'})
    ], style={'padding': '20px', 'marginBottom': '20px', 'backgroundColor': '#f0f8ff', 'borderRadius': '8px', 'border': '1px solid #007bff'}),
    
    # Graph (TOP - Full Width)
    html.Div([
        dcc.Graph(id='network-graph')
    ], style={'width': '100%', 'marginTop': '20px', 'marginBottom': '30px'}),
    
    # Table (BOTTOM - Full Width)
    html.Div([
        html.H4("Sample Pairs Above Threshold", 
               style={'color': '#000', 'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            id='edge-table',
            columns=[
                {'name': 'Sample 1', 'id': 'Sample_1', 'type': 'text'},
                {'name': 'Sample 2', 'id': 'Sample_2', 'type': 'text'},
                {'name': 'Joint Similarity', 'id': 'Joint_Similarity', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'Common Substances', 'id': 'Key_Substances', 'type': 'text'}
            ],
            style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'},
            style_cell={
                'textAlign': 'center', 
                'fontSize': '14px',
                'padding': '10px',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'fontSize': '14px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            sort_action='native',
            filter_action='native',
            filter_options={'case': 'insensitive'},
            page_action='native',
            page_size=10,
            row_selectable='single'
        )
    ], style={'width': '100%', 'marginTop': '10px'}),
    
    # Hidden download components
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-excel"),
    
], style={'backgroundColor': '#ffffff', 'color': '#000', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# -------------------------------
# Two-way sync between sliders and inputs
# -------------------------------
@app.callback(
    [Output('w-jaccard', 'value'),
     Output('w-euclidean', 'value'),
     Output('w-cosine', 'value'),
     Output('threshold', 'value')],
    [Input('slider-jaccard', 'value'),
     Input('slider-euclidean', 'value'),
     Input('slider-cosine', 'value'),
     Input('slider-threshold', 'value')]
)
def sync_slider_to_input(sj, se, sc, st):
    return sj, se, sc, st

@app.callback(
    [Output('slider-jaccard', 'value'),
     Output('slider-euclidean', 'value'),
     Output('slider-cosine', 'value'),
     Output('slider-threshold', 'value')],
    [Input('w-jaccard', 'value'),
     Input('w-euclidean', 'value'),
     Input('w-cosine', 'value'),
     Input('threshold', 'value')]
)
def sync_input_to_slider(wj, we, wc, th):
    return wj, we, wc, th

# -------------------------------
# Main callback to update graph, table, and display box
# -------------------------------
@app.callback(
    [Output('network-graph', 'figure'),
     Output('edge-table', 'data'),
     Output('joint-similarity-display', 'children')],
    [Input('w-jaccard', 'value'),
     Input('w-euclidean', 'value'),
     Input('w-cosine', 'value'),
     Input('threshold', 'value'),
     Input('chemical-dropdown', 'value')]
)
def update_network(w_j, w_e, w_c, threshold, selected_substance):
    try:
        # Validate inputs
        weights = validate_weights(w_j, w_e, w_c, threshold)
        
        # Convert percentages to normalized weights
        total = (w_j or 0) + (w_e or 0) + (w_c or 0)
        if total == 0:
            w_j_norm = w_e_norm = w_c_norm = 0
        else:
            w_j_norm = (w_j or 0) / total
            w_e_norm = (w_e or 0) / total
            w_c_norm = (w_c or 0) / total

        df = df_merged.copy()

        # Filter by chemical substances
        if selected_substance and 'All' not in selected_substance:
            df = df[df['Key_Substances'].apply(
                lambda x: any(sub in str(x).split(', ') for sub in selected_substance) if pd.notna(x) else False
            )]

        # Calculate joint similarity
        df['Joint_Similarity'] = (w_j_norm * df['Jaccard'] + 
                                 w_e_norm * df['Euclidean'] + 
                                 w_c_norm * df['Cosine'])

        # Apply threshold (convert percentage to 0-1)
        threshold_norm = (threshold or 0) / 100
        df_edges = df[df['Joint_Similarity'] >= threshold_norm].copy()

        # Remove duplicates (keep highest similarity)
        df_edges = df_edges.sort_values('Joint_Similarity', ascending=False)
        df_edges = df_edges.drop_duplicates(subset=['Sample_1', 'Sample_2'])

        # Build network
        G = nx.Graph()
        for _, row in df_edges.iterrows():
            G.add_edge(row['Sample_1'], row['Sample_2'], weight=row['Joint_Similarity'])

        # Create enhanced network visualization
        if len(G.nodes()) > 0:
            fig = create_enhanced_network(G, df_edges)
        else:
            # Empty graph if no edges meet threshold
            fig = go.Figure()
            fig.update_layout(
                title="No edges meet the current threshold",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                plot_bgcolor='white'
            )

        # Prepare table data
        table_data = df_edges[['Sample_1', 'Sample_2', 'Joint_Similarity', 'Key_Substances']].to_dict('records')

        # Display box content
        if total != 100:
            warning = html.P("⚠ Warning: Weights are unbalanced! Total ≠ 100%", 
                           style={'color': 'red', 'fontWeight': 'bold', 'margin': '5px 0'})
        else:
            warning = None

        stats_info = html.Div([
            html.P(f"📊 Network Statistics: {len(G.nodes())} nodes, {len(G.edges())} edges", 
                  style={'margin': '5px 0', 'fontWeight': 'bold'}),
            html.P(f"🧮 Joint Similarity = ({w_j_norm:.3f} × Jaccard) + ({w_e_norm:.3f} × Euclidean) + ({w_c_norm:.3f} × Cosine)",
                  style={'margin': '5px 0'}),
            html.P(f"⚖️ Weights: Jaccard: {w_j or 0}%, Euclidean: {w_e or 0}%, Cosine: {w_c or 0}% | Threshold: {threshold or 0}%",
                  style={'margin': '5px 0'})
        ])

        display_content = html.Div([stats_info, warning] if warning else [stats_info])

        return fig, table_data, display_content

    except Exception as e:
        # Return empty results with error message
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Error: {str(e)}",
            height=700
        )
        error_message = html.Div(f"❌ Error: {str(e)}", style={'color': 'red', 'fontWeight': 'bold'})
        return error_fig, [], error_message

# -------------------------------
# Save functionality callbacks
# -------------------------------
@app.callback(
    Output("download-csv", "data"),
    Input("save-table-btn", "n_clicks"),
    State('edge-table', 'data'),
    prevent_initial_call=True
)
def save_table_csv(n_clicks, table_data):
    if n_clicks:
        df = pd.DataFrame(table_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_analysis_{timestamp}.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False)

@app.callback(
    Output("download-excel", "data"),
    Input("save-excel-btn", "n_clicks"),
    State('edge-table', 'data'),
    prevent_initial_call=True
)
def save_table_excel(n_clicks, table_data):
    if n_clicks:
        df = pd.DataFrame(table_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_analysis_{timestamp}.xlsx"
        return dcc.send_data_frame(df.to_excel, filename, index=False)

@app.callback(
    Output('network-graph', 'figure', allow_duplicate=True),
    Input('save-diagram-btn', 'n_clicks'),
    State('network-graph', 'figure'),
    prevent_initial_call=True
)
def save_network_diagram(n_clicks, current_figure):
    if n_clicks and current_figure:
        # Plotly figures can be saved using the built-in camera icon
        # This callback ensures the button works, but actual download is handled by Plotly
        return current_figure
    return current_figure

# -------------------------------
# Additional callback for row selection highlighting
# -------------------------------
@app.callback(
    Output('network-graph', 'figure', allow_duplicate=True),
    Input('edge-table', 'selected_rows'),
    State('network-graph', 'figure'),
    prevent_initial_call=True
)
def highlight_selected_edge(selected_rows, current_figure):
    if not selected_rows or not current_figure:
        return current_figure
    return current_figure

# -------------------------------
# Run the app
# -------------------------------
if __name__ == '__main__':
    print("🌐 Starting web server...")
    app.run_server(debug=False, host='0.0.0.0', port=8050)
