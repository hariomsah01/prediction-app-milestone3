import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import base64
import io

app = dash.Dash(__name__)
server = app.server

df_global = None
model = None
features_used = []
target_column = ""

app.layout = html.Div([
    html.H4("Upload File"),
    dcc.Upload(id='upload-data', children=html.Div(["📁 Drag and Drop or Click to Upload CSV"]), style={
        'width': '100%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
        'textAlign': 'center', 'marginBottom': '20px'
    }),
    html.Div(id='file-info'),

    html.Label("Select Target:"),
    dcc.Dropdown(id='target-dropdown', placeholder="Select a numeric target variable"),

    html.Br(),
    html.Label("Select Categorical Variable:"),
    dcc.RadioItems(id='cat-radio', inline=True),

    html.Div([
        html.Div([dcc.Graph(id='bar-cat')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bar-corr')], style={'width': '48%', 'display': 'inline-block'})
    ]),

    html.Br(),
    html.Label("Select Features:"),
    dcc.Checklist(id='feature-checklist', inline=True),

    html.Br(),
    html.Button("Train", id="train-btn", style={'marginBottom': '10px'}),
    html.Div(id="train-output"),

    html.Div([
        html.Div(id='expected-format', style={'marginBottom': '5px'}),
        dcc.Input(id='predict-input', type='text', placeholder='Enter values matching selected features'),
        html.Button('Predict', id='predict-btn', n_clicks=0, style={'marginLeft': '10px'}),
        html.Span(id='predict-output', style={'marginLeft': '10px'})
    ])
], style={'width': '90%', 'margin': 'auto', 'fontFamily': 'Arial'})

@app.callback(
    Output('file-info', 'children'),
    Output('target-dropdown', 'options'),
    Output('target-dropdown', 'value'),
    Output('cat-radio', 'options'),
    Output('cat-radio', 'value'),
    Output('feature-checklist', 'options'),
    Output('feature-checklist', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(contents, filename):
    global df_global
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_global = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        num_cols = df_global.select_dtypes(include='number').columns.tolist()
        cat_cols = df_global.select_dtypes(include='object').columns.tolist()
        return (
            f"Loaded `{filename}` — shape: {df_global.shape}",
            [{'label': col, 'value': col} for col in num_cols],
            num_cols[0] if num_cols else None,
            [{'label': col, 'value': col} for col in cat_cols],
            cat_cols[0] if cat_cols else None,
            [{'label': col, 'value': col} for col in df_global.columns],
            df_global.columns.tolist()
        )
    return "", [], None, [], None, [], []

@app.callback(
    Output('bar-cat', 'figure'),
    Output('bar-corr', 'figure'),
    Input('target-dropdown', 'value'),
    Input('cat-radio', 'value')
)
def update_graphs(target, cat_col):
    if df_global is None or not target or not cat_col:
        return {}, {}
    avg_df = df_global.groupby(cat_col)[target].mean().reset_index()
    fig1 = px.bar(avg_df, x=cat_col, y=target, title=f"Average {target} by {cat_col}")
    fig1.update_traces(marker_color='lightblue', texttemplate='%{y:.2f}', textposition='outside')
    fig1.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis_title=f"{target} (Average)")

    corr = df_global.select_dtypes(include='number').corr()[[target]].abs().drop(target).sort_values(by=target, ascending=False)
    fig2 = px.bar(corr, x=corr.index, y=target,
                  title=f"Correlation Strength of Numerical Variables with {target}",
                  labels={target: "Correlation Strength (Absolute Value)"})
    fig2.update_traces(marker_color='royalblue', texttemplate='%{y:.2f}', textposition='outside')
    fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis_title="Correlation Strength")

    return fig1, fig2

@app.callback(
    Output('train-output', 'children'),
    Input('train-btn', 'n_clicks'),
    State('target-dropdown', 'value'),
    State('feature-checklist', 'value')
)
def train_model(n, target, features):
    global model, features_used, target_column
    if df_global is None or not features:
        return "No data or features selected."
    X = df_global[features]
    y = df_global[target]
    cat_feats = X.select_dtypes(include='object').columns.tolist()
    num_feats = X.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    features_used = features
    target_column = target
    return f"The R2 score is: {r2:.2f}"

@app.callback(
    Output('predict-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('predict-input', 'value')
)
def make_prediction(n_clicks, input_str):
    global features_used
    if model is None or not input_str:
        return ""
    try:
        raw_values = [v.strip() for v in input_str.split(",")]
        parsed = [float(v) if v.replace('.', '', 1).isdigit() else v for v in raw_values]
        input_df = pd.DataFrame([parsed], columns=features_used)
        pred = model.predict(input_df)
        return f"Predicted {target_column} is : {pred[0]:.2f}"
    except ValueError as e:
        return f"⚠️ Input Error: {str(e)}"
    except Exception as e:
        return f"❌ Failed to predict: {str(e)}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run_server(debug=True)

@app.callback(
    Output('expected-format', 'children'),
    Input('feature-checklist', 'value')
)
def update_format_list(selected_features):
    if selected_features:
        return html.Div(f"📝 Enter values for: {', '.join(selected_features)}")
    return ""
