import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Upload App", layout="wide")

st.title("CSV Upload App")

with st.sidebar:
    sales_file = st.file_uploader("Upload sales.csv", type=["csv"], key="sales")
    inv_file = st.file_uploader("Upload inventory.csv", type=["csv"], key="inv")
    tgt_file = st.file_uploader("Upload target.csv", type=["csv"], key="tgt")
    run = st.button("Run")
    download_placeholder = st.empty()


def read_csv_safe(uploaded_file):
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, required_cols: list[str], file_label: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} missing required columns: {', '.join(missing)}")


if "result_df" not in st.session_state:
    st.session_state.result_df = None

if run:
    try:
        if sales_file is None or inv_file is None or tgt_file is None:
            st.session_state.result_df = None
            st.error("Please upload sales.csv, inventory.csv, and target.csv first.")
        else:
            sales = read_csv_safe(sales_file)
            inv = read_csv_safe(inv_file)
            tgt = read_csv_safe(tgt_file)

            validate_columns(sales, ["SKU", "Outlet", "SalesQty"], "sales.csv")
            validate_columns(inv, ["SKU", "Outlet"], "inventory.csv")
            validate_columns(tgt, ["SKU"], "target.csv")

            result = (
                sales
                .merge(inv, on=["SKU", "Outlet"], how="left")
                .merge(tgt, on=["SKU"], how="left")
            )

            st.session_state.result_df = result
    except Exception as e:
        st.session_state.result_df = None
        st.error(f"Error: {e}")

if st.session_state.result_df is not None:
    output = st.session_state.result_df
    st.table(output.head(20))
    st.text(f"Done. Rows: {len(output)}")

    csv_bytes = output.to_csv(index=False).encode("utf-8-sig")
    with st.sidebar:
        st.download_button(
            "Download Result",
            data=csv_bytes,
            file_name="result.csv",
            mime="text/csv",
        )
else:
    st.table(pd.DataFrame())
    st.text("")
