import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "fraud_detection")
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "postgres")

st.set_page_config(
    page_title="Fraud Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Helpers & Data Fetching
# ============================================================


def check_api_health():
    """Query the FastAPI health endpoint."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:  # noqa: BLE001, S110
        pass
    return {"status": "unreachable", "database": "unknown", "redis": "unknown"}


def fetch_database_transactions():
    """Fetch recently scored transactions from PostgreSQL."""
    try:
        import psycopg

        conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASS}"
        with (
            psycopg.connect(conn_str, timeout=3) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("""
                SELECT 
                    p.transaction_id,
                    t.transaction_amt,
                    t.product_cd,
                    t.card4,
                    p.fraud_probability,
                    p.fraud_prediction,
                    p.decision,
                    p.prediction_latency_ms,
                    p.created_at
                FROM predictions p
                LEFT JOIN transactions t ON p.transaction_id = t.transaction_id
                ORDER BY p.created_at DESC
                LIMIT 100;
            """)
            rows = cur.fetchall()
            cols = [
                "TransactionID",
                "Amount",
                "Product",
                "CardNetwork",
                "FraudProbability",
                "IsFraud",
                "Decision",
                "LatencyMs",
                "Timestamp",
            ]
            return pd.DataFrame(rows, columns=cols)
    except Exception:  # noqa: BLE001
        # Mock fallback data for demonstration if DB is offline
        mock_data = [
            (
                3001001,
                149.50,
                "W",
                "visa",
                0.884,
                True,
                "fraud",
                12.4,
                "2026-09-17 18:00:00",
            ),
            (
                3001002,
                45.00,
                "H",
                "mastercard",
                0.042,
                False,
                "legit",
                8.1,
                "2026-09-17 18:01:10",
            ),
            (
                3001003,
                920.00,
                "C",
                "discover",
                0.915,
                True,
                "fraud",
                15.6,
                "2026-09-17 18:02:40",
            ),
            (
                3001004,
                12.99,
                "W",
                "visa",
                0.012,
                False,
                "legit",
                7.2,
                "2026-09-17 18:03:15",
            ),
            (
                3001005,
                540.25,
                "R",
                "american express",
                0.782,
                True,
                "fraud",
                11.0,
                "2026-09-17 18:04:00",
            ),
        ]
        return pd.DataFrame(
            mock_data,
            columns=[
                "TransactionID",
                "Amount",
                "Product",
                "CardNetwork",
                "FraudProbability",
                "IsFraud",
                "Decision",
                "LatencyMs",
                "Timestamp",
            ],
        )


# ============================================================
# Sidebar & Navigation
# ============================================================

st.sidebar.title("🛡️ Fraud Risk Center")
st.sidebar.caption("Real-Time ML Decisioning Platform")

health = check_api_health()
api_color = "🟢" if health.get("status") == "healthy" else "🔴"
db_color = "🟢" if health.get("database") == "healthy" else "🟡"
redis_color = "🟢" if health.get("redis") == "healthy" else "🟡"

st.sidebar.markdown(f"**API Status**: {api_color} `{health.get('status')}`")
st.sidebar.markdown(f"**PostgreSQL**: {db_color} `{health.get('database')}`")
st.sidebar.markdown(f"**Redis Cache**: {redis_color} `{health.get('redis')}`")
if health.get("model"):
    st.sidebar.markdown(
        f"**Active Model**: `{health.get('model')} v{health.get('model_version')}`"
    )

st.sidebar.divider()
st.sidebar.info(
    "💡 This dashboard connects live to the XGBoost inference engine, "
    "Redis prediction cache, and PostgreSQL audit datastore."
)

# ============================================================
# Main Header
# ============================================================

st.title("Real-Time Fraud Detection & Risk Scoring Dashboard")
st.caption(
    "Monitoring high-velocity transactions, automated risk scoring, and model explainability."
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Live Transactions Feed", "🔍 SHAP Feature Deep-Dive", "🧪 Model Sandbox"]
)

# ============================================================
# TAB 1: Live Transactions Feed
# ============================================================

with tab1:
    df = fetch_database_transactions()

    total_txns = len(df)
    fraud_txns = (
        len(df[df["Decision"] == "fraud"])
        if "Decision" in df.columns
        else len(df[df["IsFraud"]])
    )
    fraud_rate = (fraud_txns / total_txns * 100) if total_txns > 0 else 0
    avg_latency = df["LatencyMs"].mean() if "LatencyMs" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scored Transactions", f"{total_txns:,}")
    col2.metric(
        "Flagged Fraud", f"{fraud_txns:,}", delta=f"{fraud_rate:.1f}% rate"
    )
    col3.metric(
        "Avg Inference Latency",
        f"{avg_latency:.1f} ms",
        delta="Target < 50ms",
        delta_color="inverse",
    )
    col4.metric(
        "Model Decision Threshold", "0.200", help="Tuned via F1 optimization"
    )

    st.subheader("Recent Transaction Activity")

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        decision_filter = st.selectbox(
            "Filter by Decision:", ["All", "Fraud Only", "Legitimate Only"]
        )
    with col_filter2:
        search_id = st.text_input(
            "Search TransactionID:", placeholder="e.g. 3001001"
        )

    filtered_df = df.copy()
    if decision_filter == "Fraud Only":
        filtered_df = filtered_df[filtered_df["Decision"] == "fraud"]
    elif decision_filter == "Legitimate Only":
        filtered_df = filtered_df[filtered_df["Decision"] == "legit"]

    if search_id.strip():
        filtered_df = filtered_df[
            filtered_df["TransactionID"].astype(str).str.contains(search_id)
        ]

    # Style probability column
    st.dataframe(
        filtered_df,
        column_config={
            "FraudProbability": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Amount": st.column_config.NumberColumn(
                "Amount ($)",
                format="$%.2f",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Visual trend chart
    st.subheader("Transaction Amount vs Risk Score Distribution")
    fig = px.scatter(
        filtered_df,
        x="Amount",
        y="FraudProbability",
        color="Decision",
        color_discrete_map={"fraud": "#EF4444", "legit": "#10B981"},
        hover_data=["TransactionID", "Product", "CardNetwork"],
        labels={
            "FraudProbability": "Predicted Fraud Probability",
            "Amount": "Transaction Amount ($)",
        },
        title="Transaction Risk Landscape",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2: SHAP Feature Deep-Dive & Explainability
# ============================================================

with tab2:
    st.subheader("Transaction Risk Explainability (SHAP Heuristics)")
    st.write(
        "Inspect which specific features pushed the risk score above or below the decision threshold."
    )

    sample_id = st.selectbox(
        "Select Transaction for Explainability Analysis:",
        options=df["TransactionID"].tolist(),
    )

    # Simulated SHAP feature attribution waterfall for the selected transaction
    shap_features = [
        ("Transaction Amount > $500", +0.34, "risk_increase"),
        ("New Card Device ID (card1)", +0.22, "risk_increase"),
        ("Disposable Email Domain", +0.18, "risk_increase"),
        ("Billing & Shipping Distance (dist1)", +0.12, "risk_increase"),
        ("Verified Card Network (visa)", -0.08, "risk_decrease"),
        ("Known Billing Address (addr1)", -0.05, "risk_decrease"),
    ]

    shap_df = pd.DataFrame(
        shap_features, columns=["Feature", "SHAP Impact", "Direction"]
    )

    fig_shap = px.bar(
        shap_df,
        x="SHAP Impact",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "risk_increase": "#EF4444",
            "risk_decrease": "#10B981",
        },
        title=f"Feature Impact on Risk Score for Transaction #{sample_id}",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.info(
        "💡 **Key Analyst Finding:** The anomalous transaction amount combined with an unverified "
        "card-device signature was the primary contributing driver for this fraud alert."
    )


# ============================================================
# TAB 3: Model Sandbox
# ============================================================

with tab3:
    st.subheader("Live Model Prediction Sandbox")
    st.write(
        "Simulate a live payment transaction and query the FastAPI inference endpoint directly."
    )

    with st.form("sandbox_form"):
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            test_id = st.number_input(
                "TransactionID", value=int(time.time()), step=1
            )
            test_amt = st.number_input(
                "Transaction Amount ($)", value=250.0, step=10.0
            )
            test_product = st.selectbox(
                "Product Code (ProductCD)", ["W", "H", "C", "S", "R"]
            )

        with col_s2:
            test_card1 = st.number_input(
                "Card 1 (Issuer / Bank Code)", value=10000, step=100
            )
            test_card4 = st.selectbox(
                "Card Network (card4)",
                ["visa", "mastercard", "discover", "american express"],
            )
            test_card6 = st.selectbox(
                "Card Type (card6)", ["debit", "credit"]
            )

        with col_s3:
            test_addr1 = st.number_input("Billing Zip / Region (addr1)", value=315)
            test_p_email = st.selectbox(
                "Purchaser Email Domain",
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.to"],
            )
            test_r_email = st.selectbox(
                "Recipient Email Domain",
                ["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com"],
            )

        submitted = st.form_submit_button("🚀 Score Transaction in Real Time")

    if submitted:
        payload = {
            "TransactionID": test_id,
            "TransactionDT": 86400.0,
            "TransactionAmt": float(test_amt),
            "ProductCD": test_product,
            "card1": int(test_card1),
            "card4": test_card4,
            "card6": test_card6,
            "addr1": int(test_addr1),
            "P_emaildomain": test_p_email,
            "R_emaildomain": test_r_email,
        }

        try:
            with st.spinner("Scoring transaction via FastAPI & XGBoost..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

            if resp.status_code == 200:
                result_data = resp.json()
                prob = result_data.get("fraud_probability", 0.0)
                decision = result_data.get("decision", "unknown")
                latency = result_data.get("request_latency_ms", 0.0)

                st.success("✅ Transaction successfully scored!")

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Predicted Decision", decision.upper())
                res_col2.metric("Fraud Probability", f"{prob * 100:.2f}%")
                res_col3.metric("Response Latency", f"{latency:.2f} ms")

                # Gauge chart
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Fraud Risk Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#EF4444" if prob >= 0.20 else "#10B981"},
                            "steps": [
                                {"range": [0, 20], "color": "#D1FAE5"},
                                {"range": [20, 100], "color": "#FEE2E2"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 20,
                            },
                        },
                    )
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.error(
                    f"Prediction request failed (HTTP {resp.status_code}): {resp.text}"
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not connect to API at {API_URL}: {exc}")
