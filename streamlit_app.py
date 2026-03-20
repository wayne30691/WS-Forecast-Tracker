import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Upload App", layout="wide")

st.title("CSV Upload App")

with st.sidebar:
    st.header("Upload files")
    sales_file = st.file_uploader("Upload sales.csv", type=["csv"], key="sales")
    inv_file = st.file_uploader("Upload inventory.csv", type=["csv"], key="inv")
    tgt_file = st.file_uploader("Upload target.csv", type=["csv"], key="tgt")
    run = st.button("Run", use_container_width=True)

def read_csv_safe(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    try:
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
            st.success(f"Done. Rows: {len(result):,}")

    except Exception as e:
        st.session_state.result_df = None
        st.error(f"Error: {e}")

st.subheader("Preview")

if st.session_state.result_df is not None:
    preview_df = st.session_state.result_df.head(20)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    csv_data = st.session_state.result_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download Result",
        data=csv_data,
        file_name="result.csv",
        mime="text/csv",
    )
else:
    st.info("Upload the three CSV files and click Run.")
