#carga de librerias para machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    confusion_matrix,
                    mean_absolute_error,
                    mean_squared_error,
                    r2_score,
                    roc_curve,
                    auc
                    )
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import label_binarize

import plotly.graph_objects as go


# Carga de librerias para dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.express as px



# cargue de datos
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
filas = df.shape[0]
columnas = df.shape[1]

#preparación de datos para machine Learning
#Modelo logistico
X = df.drop("species", axis=1)
y = df["species"]
#Entrenamiento del Modelo
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train,y_train)
y_pred = modelo.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred),3)
precision = round(precision_score(y_test,y_pred,average="weighted"),3)
recall = round(recall_score(y_test,y_pred,average="weighted" ),3)
f1 = round(f1_score(y_test,y_pred,average="weighted"),3)

card_accuracy = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Accuracy",className="text-white"),
                    html.H2(str(accuracy),className="text-white"),
                    ]
                ),
                color="primary"
                )

card_precision = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Precision",className="text-white"),
                    html.H2(str(precision),className="text-white")
                    ]
                ),
                color="success"
                )

card_recall = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Recall",className="text-white"),
                    html.H2(str(recall),className="text-white")
                    ]
                ),
                color="warning"
                )

card_f1 = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("F1 Score",className="text-white"),
                    html.H2(str(f1),className="text-white")
                    ]
                ),
                color="danger"
                )



#generación de la matriz de confusión
cm = confusion_matrix(y_test,y_pred)

heatmap_confusion = px.imshow(cm,text_auto=True,color_continuous_scale="Blues",
                              x=modelo.classes_,y=modelo.classes_,
                              title="Matriz de Confusión"
                            )
heatmap_confusion.update_layout(
                    title_x=0.5,
                    template="plotly_white",
                    xaxis_title="Predicción",
                    yaxis_title="Valor Real"
                    )

#machine learning
#Modelo lineal
X_reg = df[["sepal_length","sepal_width","petal_width"]]
y_reg = df["petal_length"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
X_reg,
y_reg,
test_size=0.2,
random_state=42
)

modelo_rl = LinearRegression()
modelo_rl.fit(
X_train_reg,
y_train_reg
)

y_pred_reg = modelo_rl.predict(X_test_reg)

# preparación de la curva AUC ROC
y_score = modelo.predict_proba(X_test)
clases = modelo.classes_
y_test_bin = label_binarize(y_test,classes=clases)

#Calculo de la curva AUC ROC
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(clases)):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc[i] = auc(fpr[i],tpr[i])
    
fig_roc = go.Figure()
for i, clase in enumerate(clases):
    fig_roc.add_trace(go.Scatter(x=fpr[i],y=tpr[i],mode="lines",name=f"{clase} (AUC={roc_auc[i]:.2f})"))

fig_roc.add_trace(go.Scatter(x=[0, 1],y=[0, 1],mode="lines",line=dict(dash="dash",color="gray"), name="Azar"))
fig_roc.update_layout(title="Curva ROC - One vs Rest",
                      xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        template="plotly_white",
                        title_x=0.5
                        )

   

r2 = round(r2_score(y_test_reg,y_pred_reg),3)
mae = round(mean_absolute_error(y_test_reg,y_pred_reg),3)
rmse = round(mean_squared_error(y_test_reg,y_pred_reg) ** 0.5,3)

fig_regresion = px.scatter(x=y_test_reg,y=y_pred_reg,
                labels={
                        "x": "Valor Real",
                        "y": "Predicción"
                        },
                    title="Regresión Lineal: Real vs Predicción",
                    opacity=0.8
                )

fig_regresion.add_shape(type="line",
                            x0=min(y_test_reg),
                            y0=min(y_test_reg),
                            x1=max(y_test_reg),
                            y1=max(y_test_reg),
                            line=dict(
                            color="red",
                            width=3,
                            dash="dash"
                            )
                        )
fig_regresion.update_layout(title_x=0.5)

fig_regresion.update_traces(
                        marker=dict(
                        size=12,
                        color="#3498db",
                        line=dict(
                        width=1,
                        color="black"
                        )
                        )
                )

fig_regresion.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"R² = {r2}",
                        showarrow=False,
                        font=dict(
                        size=14
                        ),
                        bgcolor="white"
                        )




card_r2 = dbc.Card(dbc.CardBody(
                [
                html.H5("R²"),
                html.H2(
                str(r2),
                className="text-primary"
                )
                ]
                ),
                color="light",
                className="shadow rounded-4"
                )

card_mae = dbc.Card(
    dbc.CardBody(
        [
            html.H6(
                "MAE",
                className="text-white"
            ),

            html.H2(
                str(mae),
                className="text-white"
            )
        ],
        className="text-center"
    ),
    color="success",
    className="shadow rounded-4"
)

card_rmse = dbc.Card(
    dbc.CardBody(
        [
            html.H6(
                "RMSE",
                className="text-white"
            ),

            html.H2(
                str(rmse),
                className="text-white"
            )
        ],
        className="text-center"
    ),

    color="danger",
    className="shadow rounded-4"
)


#Indicadores clave
total_registros = len(df)
total_variables = len(df.columns)
total_especies = df["species"].nunique()
promedio_petal_length = round(df["petal_length"].mean(),2)
promedio_petal_width = round(df["petal_width"].mean(),2)
promedio_sepal_length = round(df["sepal_length"].mean(),2)
promedio_sepal_width = round(df["sepal_width"].mean(),2)

card_registros = dbc.Card(
                    dbc.CardBody(
                        [
                        html.H5("Total Registros"),
                        html.H2(
                        str(total_registros),
                        className="text-primary"
                        )
                        ]
                    )
                )

card_registros = dbc.Card(
                    dbc.CardBody(
                        [
                        html.H5("Total Registros"),
                        html.H2(
                        total_registros,
                        className="text-primary"
                        )
                        ]
                    ),
                    className="shadow"
                )

card_variables = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Variables"),
                    html.H2(
                    total_variables,
                    className="text-success"
                    )
                    ]
                ),
                className="shadow"
                )
card_especies = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Especies"),
                    html.H2(
                    total_especies,
                    className="text-danger"
                    )
                    ]
                ),
                className="shadow"
                )

card_petal_length = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Promedio Petal Length"),
                    html.H2(
                    promedio_petal_length,
                    className="text-warning"
                    )
                    ]
                ),
                className="shadow"
                )

card_petal_width = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Promedio Petal Width"),
                    html.H2(
                    promedio_petal_width,
                    className="text-info"
                    )
                    ]
                ),
                className="shadow"
                )

card_sepal_length = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Promedio Sepal Length"),
                    html.H2(
                    promedio_sepal_length,
                    className="text-primary"
                    )
                    ]
                ),
                className="shadow"
                )

card_sepal_width = dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Promedio Sepal Width"),
                    html.H2(
                    promedio_sepal_width,
                    className="text-secondary"
                    )
                    ]
                ),
                className="shadow"
                )
# gráficas para PKI
grafico_especies = px.pie(
                df,
                names="species",
                title="Distribución de Especies",
                hole=0.4,color_discrete_sequence=[
                            "#3498db",
                            "#2ecc71",
                            "#e74c3c"
                            ]
                )
grafico_especies.update_traces(
textinfo="percent+label",
pull=[0.05, 0.05, 0.05]
)
grafico_especies.update_layout(
title={"text": "Distribución de las Especies Iris","x": 0.5},
template="plotly_white"
)



#variables de análisis
variables_numericas = df.select_dtypes(
include="number"
).columns.tolist()

# matriz de correlacion con mapa de calor
corr = df[variables_numericas].corr()
heatmap = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Matriz de Correlación"
                )



#gráficas para el EDA
histograma = px.histogram(
            df,
            x="sepal_length",
            color="species",
            title="Distribución de Longitud de sepalo",
            nbins=20,
            )

boxplot= px.box(
            df,
            x="species",
            y="sepal_length",
            color="species",
            title="Longitud de petalo por especie"
            )

scatter = px.scatter(
            df,
            x="petal_length",
            y="petal_width",
            color="species",
            title="Longitud de petalo vs Ancho Petalo"
            )


#aplicación
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY]
)

app.layout = dbc.Container(
    [
        html.H1(
            "Dashboard de Ciencia de Datos - Iris",
            className="text-center my-4"
        ),

        html.P(
            "Análisis Exploratorio, KPIs y Regresión Logística",
            className="text-center text-muted"
            ),
        
        html.P(
                "Juan Carlos Alarcon - Prog. Ciencia datos II",
                className="text-center text-muted"
                ),
        
        dcc.Tabs(
            style={
                    "marginTop": "10px"
                },
            children=[
                
#-----  Análisis EDA
                dcc.Tab(                  
                    label="EDA",
                    children=[
                        html.Br(),
                        html.H3("Análisis Exploratorio de Datos"),
                        html.P(f"El dataset contiene {filas} registros y {columnas} variables."),
                        html.Hr(),
                        html.P(f"Especies encontradas: {df['species'].nunique()}"),
                        html.H4("Primeras observaciones"),
                        dash_table.DataTable(
                            data=df.head().to_dict("records"),
                            columns=[
                                    {"name": col, "id": col}
                                    for col in df.columns
                                    ],
                                    page_size=5,
                                style_table={"overflowX":"auto"}, 
                                style_header={"fontWeight":"bold"},
                                style_cell={"textAlign":"center"}   
                                ),
                        html.Hr(),
                        
                        html.H4("Resumen Estadístico"),
                        dash_table.DataTable(
                            data=df.describe().round(2).reset_index().to_dict("records"),
                            columns=[
                                    {"name": col, "id": col}
                                    for col in df.describe().reset_index().columns
                                    ],
                            style_table={"overflowX":"auto"}, 
                            style_header={"fontWeight":"bold"},
                            style_cell={"textAlign":"center"} 
                            ),
                        
                        html.Hr(),
                        html.H4("Variables del Dataset"),
                        html.Ul(
                                [
                                html.Li(col)
                                for col in df.columns
                                ]
                                ),
                      
                        html.H4("Filtrar por especie"),  
                            dcc.Dropdown(
                            id="filtro-especie",
                            options=[
                                    {"label": "Todas", "value": "all"},
                                    {"label": "Setosa", "value": "setosa"},
                                    {"label": "Versicolor", "value": "versicolor"},
                                    {"label": "Virginica", "value": "virginica"}
                                    ],
                            value="all",
                            clearable=False
                            ),
                        
                        html.Hr(),
                        html.H4("Seleccione variable:"),
                        dcc.Dropdown(
                                    id="variable-histograma",
                                    options=[
                                            {
                                            "label":col,
                                            "value":col
                                             }
                                             for col in variables_numericas
                                             ],
                                    value="sepal_length",
                                    clearable=False
                                    ),  
                                               
                        html.Hr(),
                        
                        dcc.Dropdown(id="variable-boxplot",
                                    options=[
                                    {
                                    "label": col,
                                    "value": col
                                     }
                                    for col in variables_numericas
                                    ],
                                    value="petal_length",
                                    clearable=False
                                    ),
                       
                        dbc.Row(
                                [
                                dbc.Col(
                                    dcc.Graph(id="grafico-boxplot"),
                                    md=6
                                    ),
                                
                                dbc.Col(
                                    dcc.Graph(id="grafico-histograma"),
                                    md=6
                                    ),
                                
                                dbc.Col(
                                    dcc.Graph(figure=heatmap),
                                    md=6
                                    )                   
                                ]
                                ),
                        html.Hr(),
                        dbc.Row(
                                [
                                dbc.Col(
                                    [
                                    html.Label("Variable X"),
                                    dcc.Dropdown(
                                        id="scatter-x",
                                        options=[
                                            {"label": col, "value": col}
                                            for col in variables_numericas
                                        ],
                                        value="petal_length",
                                        clearable=False
                                    )
                                    ],
                                    md=6
                                ),
                                dbc.Col(
                                    [
                                    html.Label("Variable Y"),
                                    dcc.Dropdown(
                                    id="scatter-y",
                                    options=[
                                    {"label": col, "value": col}
                                    for col in variables_numericas
                                    ],
                                    value="petal_width",
                                    clearable=False
                                    )
                                    ],
                                    md=6
                                    )
                                ]
                                ),
                        dcc.Graph(id="grafico-scatter")                                             
                    ],
                    
                    style={
                    "fontWeight": "bold"
                    },
                    selected_style={
                        "backgroundColor": "#BD430B",
                        "color": "white",
                        "fontWeight": "bold"
                    },               
                ),


#--------- Indicadores Clave
                dcc.Tab(
                    label="KPI",
                    children=[
                        html.Br(),
                        html.H3("Indicadores clave"),
                        dbc.Alert("Los KPIs permiten obtener una visión rápida y resumida del dataset Iris.",color="success"),
                        dbc.Row(
                            [
                            dbc.Col(
                                card_registros,
                                md=3
                            ),
                            dbc.Col(
                                card_variables,
                                md=3
                            ),
                            dbc.Col(
                                card_especies,
                                md=3
                                ),
                            dbc.Col(
                                card_petal_length,
                                md=3
                                )
                            ]
                        ),
                    dbc.Row(
                        [
                        dbc.Col(
                                card_sepal_length,
                                md=4
                            ),
                        dbc.Col(
                                card_sepal_width,
                                md=4
                            ),
                        dbc.Col(
                                card_petal_width,
                                md=4
                            )
                        ]
                    ),
                    html.Br(),
                    html.H4("Composición del Dataset"),
                    dcc.Graph(figure=grafico_especies)
                    ],
                    style={
                    "fontWeight": "bold"
                    },
                    selected_style={
                        "backgroundColor": "#BD430B",
                        "color": "white",
                        "fontWeight": "bold"
                     }
                ),

# modelo Machine Learning
                dcc.Tab(
                    label="Modelo de Machine Learning",
                    children=[
                        html.Br(),
                        html.H3("Regresión Logística"),
                        dbc.Alert("Clasificación automática de especies Iris utilizando Regresión Logística.",color="primary"),
                        dbc.Row(
                            [
                                dbc.Col(card_accuracy,md=3),
                                dbc.Col(card_precision,md=3),
                                dbc.Col(card_recall,md=3),
                                dbc.Col(card_f1,md=3)  
                            ]                           
                        ),
                        html.H4("Matriz de Confusión"),
                        dbc.Card(
                            dbc.CardBody(
                            [
                                html.H4("Matriz de confusion"),
                                dbc.Row(
                                    [
                                     dbc.Col(
                                         dcc.Graph(figure=heatmap_confusion),
                                         md=6
                                        ),   
                                    ]
                                )
                            ]),
                            className="shadow rounded-4"    
                        ),
                        html.H4("Interpretación"),
                        html.Ul(
                                [
                                html.Li(f"Accuracy: {accuracy}"),
                                html.Li(f"Precision: {precision}"),
                                html.Li(f"Recall: {recall}"),
                                html.Li(f"F1 Score: {f1}")
                                ]
                                ),
                        html.Br(),
                        html.H3("Regresión Lineal"),
                        dbc.Alert("Predicción de Petal Length utilizando variables numéricas del dataset Iris.",color="secondary"),
                        dbc.Row(
                            [
                                dbc.Col(card_r2,md=4),
                                dbc.Col(card_mae,md=4),
                                dbc.Col(card_rmse,md=4),
                                html.Br(),
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Valores Reales vs Predichos"),
                                            dcc.Graph(figure=fig_regresion)
                                        ]
                                    )
                                    
                                )
                                
                            ]
                        )
                        
                    ],
                    style={
                    "fontWeight": "bold"
                    },
                    selected_style={
                        "backgroundColor": "#BD430B",
                        "color": "white",
                        "fontWeight": "bold"
                     }
                )
            ]
        )
    ],
    fluid=True
)

#callbacks para cargue dinamico de información
@app.callback(
    Output("grafico-histograma", "figure"),
        [
        Input("variable-histograma", "value"),
        Input("filtro-especie", "value")
        ]
)


def actualizar_histograma(variable,especie):
        if especie == "all":
            df_filtrado = df
        else:
            df_filtrado = df[df["species"] == especie]
            
        fig = px.histogram(
            df_filtrado,
            x=variable,
            color="species",
            nbins=20,
            title=f"Distribución de {variable}"
            )
        return fig

@app.callback(
    Output("grafico-boxplot", "figure"),
    [
        Input("variable-boxplot", "value"),
        Input("filtro-especie", "value")
    ]
)
def actualizar_boxplot(variable, especie):

    if especie == "all":
        df_filtrado = df
    else:
        df_filtrado = df[df["species"] == especie]

    fig = px.box(
        df_filtrado,
        x="species",
        y=variable,
        color="species",
        title=f"{variable} por especie"
    )

    return fig

@app.callback(
    Output("grafico-scatter", "figure"),
    [
        Input("scatter-x", "value"),
        Input("scatter-y", "value"),
        Input("filtro-especie", "value")
    ]
)
def actualizar_scatter(variable_x, variable_y, especie):

    if especie == "all":
        df_filtrado = df
    else:
        df_filtrado = df[
            df["species"] == especie
        ]

    fig = px.scatter(
        df_filtrado,
        x=variable_x,
        y=variable_y,
        color="species",
        title=f"{variable_x} vs {variable_y}"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig



if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=8050,
    debug=False
    )