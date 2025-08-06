import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="스마트 팩토리 센서 분석기", layout="wide")

st.title("📊 스마트 팩토리 센서 데이터 분석기")
st.markdown("센서 데이터를 업로드하고 이상치를 감지하거나 시각화할 수 있습니다.")

uploaded_file = st.file_uploader("📁 CSV 센서 로그 또는 목록 파일 업로드", type=["csv"])

if uploaded_file:
    encodings_to_try = ['utf-8', 'cp949', 'ISO-8859-1']
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(uploaded_file, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        st.error("❗ CSV 파일을 읽는 데 실패했습니다. 파일 인코딩 또는 구조를 확인해주세요.")
        st.stop()

    if df.empty:
        st.error("❗ 업로드한 CSV 파일에 데이터가 없습니다.")
        st.stop()

    st.subheader("🔍 데이터 미리보기")
    st.dataframe(df.head())

    # 숫자형 열 추출
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        st.info("ℹ️ 이 파일에는 분석 가능한 센서 수치 데이터가 없습니다.\n\n"
                "예: 온도, 습도, 진동 등 숫자형 센서 데이터가 포함된 CSV를 업로드하면 다양한 분석 기능을 제공합니다.")
        st.stop()

    # 시계열 분석을 위한 timestamp 컬럼 존재 여부 확인
    timestamp_available = False
    if "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values("timestamp")
        timestamp_available = True

    if timestamp_available:
        st.subheader("📅 날짜 필터링")
        start_date = st.date_input("시작 날짜", value=df["timestamp"].min().date())
        end_date = st.date_input("종료 날짜", value=df["timestamp"].max().date())
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        df = df.loc[mask]

    st.subheader("📈 센서 통계 요약")
    desc_df = df[numeric_cols].describe()
    stats_kor = {
        "count": "count (개수)",
        "mean": "mean (평균)",
        "std": "std (표준편차)",
        "min": "min (최솟값)",
        "25%": "25% (1사분위)",
        "50%": "50% (중앙값)",
        "75%": "75% (3사분위)",
        "max": "max (최댓값)"
    }
    desc_df.rename(index=stats_kor, inplace=True)
    st.dataframe(desc_df)

    st.subheader("🚨 이상치 탐지 (Z-score > 3 기준)")
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    outliers = (z_scores > 3).any(axis=1)

    st.write(f"**이상치 수:** {outliers.sum()}개")
    st.dataframe(df[outliers])

    if outliers.sum() > 0:
        st.error("🚨 이상치가 감지되었습니다! 점검이 필요합니다.")
    else:
        st.success("✅ 모든 센서가 정상 범위 내에 있습니다.")

    csv_outliers = df[outliers].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 이상치 데이터 다운로드",
        data=csv_outliers,
        file_name="outliers.csv",
        mime="text/csv"
    )

    st.subheader("📊 센서별 이상치 비율")
    outlier_ratio = ((z_scores > 3).sum() / len(df)) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    outlier_ratio.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_ylabel("이상치 비율 (%)", fontsize=12)
    ax2.set_title("센서별 이상치 비율", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("🔗 센서 간 상관관계 히트맵")
    corr = df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax3.set_title("센서 간 상관관계", fontsize=14, fontweight='bold')
    fig3.tight_layout()
    st.pyplot(fig3)

    if timestamp_available:
        st.subheader("📌 센서 시계열 그래프 (이상치 강조)")
        selected_col = st.selectbox("시각화할 센서 선택", numeric_cols)

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(df['timestamp'], df[selected_col], label="정상 값", linewidth=1.5)
        ax4.scatter(df[outliers]['timestamp'], df.loc[outliers, selected_col],
                    color='red', label="이상치", zorder=5, s=40)
        ax4.set_title(f"{selected_col} 센서 값 (이상치는 빨간 점)", fontsize=14, fontweight='bold')
        ax4.set_xlabel("시간", fontsize=12)
        ax4.set_ylabel("센서 값", fontsize=12)
        ax4.legend(fontsize=10)
        fig4.autofmt_xdate(rotation=30)
        fig4.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("⏱️ 시계열 분석은 'timestamp' 컬럼이 있을 때만 제공됩니다.")

else:
    st.warning("📁 CSV 파일을 업로드해주세요.")
