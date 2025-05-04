import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("📊 Data Analysis and Prediction App")

# Upload
uploaded_file = st.file_uploader("Upload File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.write("Data Preview", df.head())

    # Select Target
    num_cols = df.select_dtypes(include='number').columns.tolist()
    target = st.selectbox("Select Target:", num_cols)

    if target:
        # Bar Charts
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            selected_cat = st.radio("Select a Categorical Variable", cat_cols)
            fig1 = px.bar(df.groupby(selected_cat)[target].mean().reset_index(), x=selected_cat, y=target,
                          title=f"Average {target} by {selected_cat}")
            st.plotly_chart(fig1)

        corr_df = df[num_cols].corr()[[target]].abs().drop(target).sort_values(by=target, ascending=False)
        fig2 = px.bar(corr_df, x=corr_df.index, y=target,
                      title=f"Correlation Strength with {target}",
                      labels={target: "Correlation Strength"})
        st.plotly_chart(fig2)

        # Train Model
        st.write("### Feature Selection")
        features = [col for col in df.columns if col != target]
        selected_features = st.multiselect("Select Features", features, default=features)

        if st.button("Train"):
            X = df[selected_features]
            y = df[target]

            cat_feats = X.select_dtypes(include='object').columns.tolist()
            num_feats = X.select_dtypes(include='number').columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), num_feats),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
            ])

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", DecisionTreeRegressor(random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            st.write(f"✅ Model trained. R² Score: {r2:.2f}")
            st.session_state.model = model
            st.session_state.features = selected_features

        # Predict
        if "model" in st.session_state:
            st.write("### Make a Prediction")
            example = ", ".join(st.session_state.features)
            user_input = st.text_input(f"Enter comma-separated values for: {example}")
            if st.button("Predict"):
                try:
                    values = [v.strip() for v in user_input.split(",")]
                    processed = [float(v) if v.replace('.', '', 1).isdigit() else v for v in values]
                    input_df = pd.DataFrame([processed], columns=st.session_state.features)
                    pred = st.session_state.model.predict(input_df)
                    st.success(f"🔮 Predicted {target}: {pred[0]:.2f}")
                except Exception as e:
                    st.error(f"Invalid input. Error: {e}")
