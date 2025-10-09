import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import os

print("Loading data...")

# -------------------------------
# Load similarity tables
# -------------------------------
try:
    df_jac = pd.read_csv("similarity_jaccard.csv")
    df_euc = pd.read_csv("similarity_euclidean_normalized.csv")
    df_cos = pd.read_csv("cosine_similarity_results.csv")
    df_substance = pd.read_csv("Joint_Similarity_Substance.csv")
    print("All CSV files loaded successfully!")
except Exception as e:
    print(f"Error loading CSV files: {e}")

# Rename similarity columns
df_jac.rename(columns={'Jaccard_Similarity':'Jaccard'}, inplace=True)
df_euc.rename(columns={'Euclidean_Similarity':'Euclidean'}, inplace=True)
df_cos.rename(columns={'Cosine_Similarity':'Cosine'}, inplace=True)

# Merge similarity tables
df_merged = df_jac.merge(df_euc, on=['Sample_1','Sample_2']).merge(df_cos, on=['Sample_1','Sample_2'])

# Aggregate substances per sample pair
df_substance = (
    df_substance.groupby(['Sample_1','Sample_2'])['Common_Substance']
    .apply(lambda x: ', '.join(sorted(set(x.dropna()))))
    .reset_index()
    .rename(columns={'Common_Substance':'Key_Substances'})
)

# Merge aggregated substance info
df_merged = df_merged.merge(df_substance, on=['Sample_1','Sample_2'], how='left')

# Generate dropdown list (unique substances)
substances = ['All'] + sorted(
    set(sum([s.split(', ') for s in df_merged['Key_Substances'].dropna()], []))
)

print(f"Data processing complete. Found {len(substances)} unique substances.")

# -------------------------------
# Initialize Dash app
# -------------------------------
app = dash.Dash(__name__)
app.title = "Joint Weighted Network GUI"
server = app.server  # Important for deployment

# -------------------------------
# Layout
# -------------------------------
def weight_input_block(label, input_id, slider_id, default_value):
    """Reusable block for weight input and slider"""
    return html.Div([
        html.Label(f"{label} (%)", style={'fontSize':'18px', 'fontWeight':'bold', 'color':'#000'}),
        dcc.Input(
            id=input_id, type='number', value=default_value, min=0, max=100, step=1,
            style={'width':'150px', 'height':'40px', 'fontSize':'18px', 'textAlign':'center', 'marginBottom':'10px'}
        ),
        dcc.Slider(
            id=slider_id, min=0, max=100, step=1, value=default_value,
            tooltip={"placement": "bottom", "always_visible": True},
            marks={i: str(i) for i in range(0, 101, 20)}
        )
    ], style={'padding':'10px'})

app.layout = html.Div([
    html.H1("Joint Weighted Network Visualization", style={'textAlign':'center', 'color':'#000'}),

    # Input + Slider blocks
    weight_input_block("Jaccard Weight", "w-jaccard", "slider-jaccard", 35),
    weight_input_block("Euclidean Weight", "w-euclidean", "slider-euclidean", 30),
    weight_input_block("Cosine Weight", "w-cosine", "slider-cosine", 35),
    weight_input_block("Threshold", "threshold", "slider-threshold", 90),

    html.Div([
        html.Label("Select Chemical Component(s):", style={'fontSize':'18px', 'fontWeight':'bold', 'color':'#000'}),
        dcc.Dropdown(
            id='chemical-dropdown',
            options=[{'label': s, 'value': s} for s in substances],
            value=['All'],
            multi=True,
            clearable=True,
            style={'fontSize':'16px'}
        )
    ], style={'padding':'10px','width':'50%'}),

    # Display box
    html.Div(id='joint-similarity-display', style={
        'border':'2px solid #888', 
        'padding':'15px', 
        'margin':'10px', 
        'backgroundColor':'#ffffff', 
        'color':'#000', 
        'fontSize':'18px',
        'fontWeight':'bold'
    }),

    dcc.Graph(id='network-graph', style={'height':'700px'}),

    html.H4("Sample pairs above threshold:", style={'color':'#000'}),
    dash_table.DataTable(
        id='edge-table',
        columns=[
            {'name':'Sample_1','id':'Sample_1'},
            {'name':'Sample_2','id':'Sample_2'},
            {'name':'Joint_Similarity','id':'Joint_Similarity'},
            {'name':'Common_Substances','id':'Key_Substances'}
        ],
        style_table={'overflowX':'auto'},
        style_cell={'textAlign':'center', 'fontSize':'16px'},
        row_selectable='single'
    )
], style={'backgroundColor':'#ffffff', 'color':'#000', 'padding':'10px'})

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
# Callback to update graph, table, and display box
# -------------------------------
@app.callback(
    [Output('network-graph','figure'),
     Output('edge-table','data'),
     Output('joint-similarity-display','children')],
    [Input('w-jaccard','value'),
     Input('w-euclidean','value'),
     Input('w-cosine','value'),
     Input('threshold','value'),
     Input('chemical-dropdown','value'),
     Input('edge-table','selected_rows')]
)
def update_network(w_j, w_e, w_c, threshold, selected_substance, selected_row):
    # Convert percentages to normalized weights
    total = (w_j or 0) + (w_e or 0) + (w_c or 0)
    if total == 0:
        w_j_norm = w_e_norm = w_c_norm = 0
    else:
        w_j_norm = (w_j or 0)/total
        w_e_norm = (w_e or 0)/total
        w_c_norm = (w_c or 0)/total

    df = df_merged.copy()

    # Filter by chemical substances
    if 'All' not in selected_substance:
        df = df[df['Key_Substances'].apply(
            lambda x: any(sub in str(x).split(', ') for sub in selected_substance)
        )]

    # Calculate joint similarity
    df['Joint_Similarity'] = w_j_norm*df['Jaccard'] + w_e_norm*df['Euclidean'] + w_c_norm*df['Cosine']

    # Apply threshold (convert percentage to 0-1)
    threshold_norm = (threshold or 0)/100
    df_edges = df[df['Joint_Similarity'] >= threshold_norm].copy()

    # Remove duplicates (keep highest)
    df_edges = df_edges.sort_values('Joint_Similarity', ascending=False).drop_duplicates(subset=['Sample_1','Sample_2'])

    # Build network
    G = nx.Graph()
    for _, row in df_edges.iterrows():
        G.add_edge(row['Sample_1'], row['Sample_2'], weight=row['Joint_Similarity'])

    pos = nx.spring_layout(G, seed=42)
    # Draw edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        mode='lines'
    )

    # Draw nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='bottom left',
        marker=dict(color='#1f78b4', size=10)
    )

    fig = go.Figure()
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(title="Weighted Similarity Network",
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    table_data = df_edges[['Sample_1','Sample_2','Joint_Similarity','Key_Substances']].to_dict('records')

    # Display box: show formula and warning if sum != 100%
    if total != 100:
        warning = html.P("⚠ Weights are unbalanced! Total ≠ 100%", style={'color':'red'})
    else:
        warning = None

    display_text = html.Div([
        html.P(f"Joint Similarity = ({w_j_norm:.2f} × Jaccard) + ({w_e_norm:.2f} × Euclidean) + ({w_c_norm:.2f} × Cosine)"),
        html.P(f"Current Weights → Jaccard: {w_j or 0}%, Euclidean: {w_e or 0}%, Cosine: {w_c or 0}%"),
        warning
    ], style={'fontSize':'18px', 'color':'#000'})

    return fig, table_data, display_text

if __name__ == '__main__':
    app.run_server(debug=True)
