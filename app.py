"""
app.py
------
Streamlit web application for the Intrusion Detection System.

Lets a user either:
  (a) fill in network traffic feature values through a form, or
  (b) pick a random sample from the test set,
and instantly see whether Decision Tree / Random Forest classify it as
NORMAL or ATTACK traffic, along with the model's confidence.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from constants import COLUMN_NAMES, PROTOCOL_TYPES, FLAG_VALUES, SERVICE_VALUES
from preprocessing import transform_with_fitted, load_raw, binarize_label

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, "..", "models")
DATA_DIR = os.path.join(THIS_DIR, "..", "data")

st.set_page_config(page_title="Intrusion Detection System", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    dt_model = joblib.load(os.path.join(MODEL_DIR, "decision_tree.joblib"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    return encoders, scaler, feature_columns, dt_model, rf_model


@st.cache_data
def load_test_samples(n=500):
    """Load a slice of the real NSL-KDD test set, for the 'try a sample' mode."""
    path = os.path.join(DATA_DIR, "KDDTest.txt")
    if not os.path.exists(path):
        return None
    df = load_raw(path)
    df = binarize_label(df)
    return df.sample(n=min(n, len(df)), random_state=None).reset_index(drop=True)


def predict_row(row_df, encoders, scaler, feature_columns, dt_model, rf_model):
    X = transform_with_fitted(row_df, encoders, scaler, feature_columns)
    dt_pred = dt_model.predict(X)[0]
    dt_proba = dt_model.predict_proba(X)[0]
    rf_pred = rf_model.predict(X)[0]
    rf_proba = rf_model.predict_proba(X)[0]
    return dt_pred, dt_proba, rf_pred, rf_proba


def render_verdict(label, model_name, pred, proba):
    is_attack = pred == 1
    confidence = proba[pred] * 100
    color = "#9C3B30" if is_attack else "#3C5B41"
    verdict = "SUSPICIOUS — possible attack" if is_attack else "SAFE — normal traffic"
    st.markdown(
        f"""
        <div style="border:1px solid {color}; border-left:5px solid {color};
                    border-radius:6px; padding:14px 18px; margin-bottom:10px;">
            <div style="font-size:13px; color:#888; text-transform:uppercase;
                        letter-spacing:0.05em;">{model_name}</div>
            <div style="font-size:20px; font-weight:600; color:{color}; margin:4px 0;">
                {verdict}
            </div>
            <div style="font-size:13px; color:#888;">Confidence: {confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    artifacts_exist = all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ["decision_tree.joblib", "random_forest.joblib",
                   "encoders.joblib", "scaler.joblib", "feature_columns.joblib"]
    )

    st.title("🛡️ Network Intrusion Detection System")
    st.caption(
        "Decision Tree & Random Forest classifiers trained on the NSL-KDD dataset — "
        "enter network traffic features and get a real-time safe / suspicious verdict."
    )

    if not artifacts_exist:
        st.error(
            "No trained models found. Run `python src/train_models.py` first "
            "to generate the model files in the `models/` folder, then reload this app."
        )
        st.stop()

    encoders, scaler, feature_columns, dt_model, rf_model = load_artifacts()

    tab1, tab2, tab3 = st.tabs(["🔍 Check traffic", "🎲 Try a real sample", "📊 Model performance"])

    # ---------------------------------------------------------------
    # TAB 1 — manual feature entry form
    # ---------------------------------------------------------------
    with tab1:
        st.subheader("Enter network traffic features")
        st.caption(
            "These are the core NSL-KDD connection features. Defaults are pre-filled "
            "with typical 'normal' traffic values — change them to see how the verdict shifts."
        )

        with st.form("traffic_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Connection basics**")
                duration = st.number_input("Duration (seconds)", min_value=0, value=0)
                protocol_type = st.selectbox("Protocol type", PROTOCOL_TYPES, index=0)
                service = st.selectbox("Service", sorted(SERVICE_VALUES), index=sorted(SERVICE_VALUES).index("http") if "http" in SERVICE_VALUES else 0)
                flag = st.selectbox("Connection flag", FLAG_VALUES, index=0)
                src_bytes = st.number_input("Bytes sent (src_bytes)", min_value=0, value=200)
                dst_bytes = st.number_input("Bytes received (dst_bytes)", min_value=0, value=1500)
                land = st.selectbox("Same host/port (land)", [0, 1], index=0)
                wrong_fragment = st.number_input("Wrong fragments", min_value=0, value=0)
                urgent = st.number_input("Urgent packets", min_value=0, value=0)

            with c2:
                st.markdown("**Content / behavior**")
                hot = st.number_input("Hot indicators", min_value=0, value=0)
                num_failed_logins = st.number_input("Failed logins", min_value=0, value=0)
                logged_in = st.selectbox("Logged in successfully", [0, 1], index=1)
                num_compromised = st.number_input("Compromised conditions", min_value=0, value=0)
                root_shell = st.selectbox("Root shell obtained", [0, 1], index=0)
                su_attempted = st.selectbox("su root attempted", [0, 1], index=0)
                num_root = st.number_input("Root accesses", min_value=0, value=0)
                num_file_creations = st.number_input("File creations", min_value=0, value=0)
                num_shells = st.number_input("Shell prompts", min_value=0, value=0)
                num_access_files = st.number_input("Access control file ops", min_value=0, value=0)
                num_outbound_cmds = st.number_input("Outbound FTP commands", min_value=0, value=0)
                is_host_login = st.selectbox("Host login", [0, 1], index=0)
                is_guest_login = st.selectbox("Guest login", [0, 1], index=0)

            with c3:
                st.markdown("**Traffic statistics (2-second window)**")
                count = st.number_input("Connections to same host", min_value=0, value=5)
                srv_count = st.number_input("Connections to same service", min_value=0, value=5)
                serror_rate = st.slider("SYN error rate", 0.0, 1.0, 0.0)
                srv_serror_rate = st.slider("Service SYN error rate", 0.0, 1.0, 0.0)
                rerror_rate = st.slider("REJ error rate", 0.0, 1.0, 0.0)
                srv_rerror_rate = st.slider("Service REJ error rate", 0.0, 1.0, 0.0)
                same_srv_rate = st.slider("Same service rate", 0.0, 1.0, 1.0)
                diff_srv_rate = st.slider("Different service rate", 0.0, 1.0, 0.0)
                srv_diff_host_rate = st.slider("Service diff host rate", 0.0, 1.0, 0.0)

            st.markdown("**Host-based traffic statistics (100-connection window)**")
            c4, c5 = st.columns(2)
            with c4:
                dst_host_count = st.number_input("Dest host count", min_value=0, max_value=255, value=255)
                dst_host_srv_count = st.number_input("Dest host service count", min_value=0, max_value=255, value=255)
                dst_host_same_srv_rate = st.slider("Dest host same service rate", 0.0, 1.0, 1.0)
                dst_host_diff_srv_rate = st.slider("Dest host diff service rate", 0.0, 1.0, 0.0)
                dst_host_same_src_port_rate = st.slider("Dest host same src port rate", 0.0, 1.0, 0.0)
            with c5:
                dst_host_srv_diff_host_rate = st.slider("Dest host service diff host rate", 0.0, 1.0, 0.0)
                dst_host_serror_rate = st.slider("Dest host SYN error rate", 0.0, 1.0, 0.0)
                dst_host_srv_serror_rate = st.slider("Dest host service SYN error rate", 0.0, 1.0, 0.0)
                dst_host_rerror_rate = st.slider("Dest host REJ error rate", 0.0, 1.0, 0.0)
                dst_host_srv_rerror_rate = st.slider("Dest host service REJ error rate", 0.0, 1.0, 0.0)

            submitted = st.form_submit_button("Analyze traffic", use_container_width=True)

        if submitted:
            row = {
                "duration": duration, "protocol_type": protocol_type, "service": service,
                "flag": flag, "src_bytes": src_bytes, "dst_bytes": dst_bytes, "land": land,
                "wrong_fragment": wrong_fragment, "urgent": urgent, "hot": hot,
                "num_failed_logins": num_failed_logins, "logged_in": logged_in,
                "num_compromised": num_compromised, "root_shell": root_shell,
                "su_attempted": su_attempted, "num_root": num_root,
                "num_file_creations": num_file_creations, "num_shells": num_shells,
                "num_access_files": num_access_files, "num_outbound_cmds": num_outbound_cmds,
                "is_host_login": is_host_login, "is_guest_login": is_guest_login,
                "count": count, "srv_count": srv_count, "serror_rate": serror_rate,
                "srv_serror_rate": srv_serror_rate, "rerror_rate": rerror_rate,
                "srv_rerror_rate": srv_rerror_rate, "same_srv_rate": same_srv_rate,
                "diff_srv_rate": diff_srv_rate, "srv_diff_host_rate": srv_diff_host_rate,
                "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
                "dst_host_same_srv_rate": dst_host_same_srv_rate,
                "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
                "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
                "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
                "dst_host_serror_rate": dst_host_serror_rate,
                "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
                "dst_host_rerror_rate": dst_host_rerror_rate,
                "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate,
            }
            row_df = pd.DataFrame([row])
            dt_pred, dt_proba, rf_pred, rf_proba = predict_row(
                row_df, encoders, scaler, feature_columns, dt_model, rf_model
            )

            st.markdown("### Result")
            r1, r2 = st.columns(2)
            with r1:
                render_verdict("traffic", "Decision Tree", dt_pred, dt_proba)
            with r2:
                render_verdict("traffic", "Random Forest", rf_pred, rf_proba)

    # ---------------------------------------------------------------
    # TAB 2 — pull a real labeled row from the NSL-KDD test set
    # ---------------------------------------------------------------
    with tab2:
        st.subheader("Try a real traffic sample from the NSL-KDD test set")
        st.caption(
            "Pulls an actual labeled connection record from the held-out test set so "
            "you can compare the model's prediction against the true label."
        )

        samples = load_test_samples()
        if samples is None:
            st.warning("Test data file not found in `data/KDDTest.txt`.")
        else:
            if st.button("Pull a random sample"):
                st.session_state["sample_idx"] = np.random.randint(0, len(samples))

            if "sample_idx" not in st.session_state:
                st.session_state["sample_idx"] = 0

            row = samples.iloc[[st.session_state["sample_idx"]]]
            true_label = "ATTACK" if row["binary_label"].values[0] == 1 else "NORMAL"
            true_raw = row["label"].values[0]

            st.dataframe(row.drop(columns=["binary_label"]), use_container_width=True)
            st.write(f"**True label:** `{true_raw}` ({true_label})")

            dt_pred, dt_proba, rf_pred, rf_proba = predict_row(
                row, encoders, scaler, feature_columns, dt_model, rf_model
            )

            r1, r2 = st.columns(2)
            with r1:
                render_verdict("traffic", "Decision Tree", dt_pred, dt_proba)
            with r2:
                render_verdict("traffic", "Random Forest", rf_pred, rf_proba)

    # ---------------------------------------------------------------
    # TAB 3 — show saved training metrics
    # ---------------------------------------------------------------
    with tab3:
        st.subheader("Model performance (NSL-KDD official test split)")
        summary_path = os.path.join(MODEL_DIR, "training_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                st.code(f.read(), language=None)
        else:
            st.info("Run `python src/train_models.py` to generate a performance summary.")

        st.caption(
            "Note: NSL-KDD's official test set intentionally includes attack patterns "
            "absent from training, so accuracy here is realistically lower than the "
            "~99% figure reported by a random train/test split of the same file. "
            "This split is the standard benchmark for this dataset."
        )


if __name__ == "__main__":
    main()
