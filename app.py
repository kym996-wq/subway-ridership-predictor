# app.py
# 🔮 지하철 이용 예측기 (Ridership Predictor)
# - 입력: 호선, 역, 날짜(→월), 시간대(필수), 요일(데이터에 있을 때만)
# - 출력: 예상 승차/하차 인원 + 과거 분포 비교 그래프/표
# - 사용법: streamlit run app.py
# - CSV 인코딩: cp949 (서울열린데이터광장 기본 배포 형식)

import os
import re
import io
import gc
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: LightGBM
LGBM_AVAILABLE = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBM_AVAILABLE = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ---------------------------
# ⚙️ Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="🔮 지하철 이용 예측기",
    page_icon="🚇",
    layout="wide"
)

st.title("🔮 지하철 이용 예측기 (Ridership Predictor)")
st.caption("서울시 지하철 호선·역·시간대별 승하차 인원 데이터 기반 | 월 단위 예측")

# ---------------------------
# 📥 데이터 로딩
# ---------------------------
st.sidebar.header("1) 데이터 불러오기")
src_opt = st.sidebar.radio("데이터 소스 선택", ["로컬 업로드", "경로 입력(/mnt 또는 GitHub Raw)"], index=0)
default_path = "/mnt/data/station.csv" if os.path.exists("/mnt/data/station.csv") else ""

uploaded_file = None
csv_bytes = None

if src_opt == "로컬 업로드":
    uploaded_file = st.sidebar.file_uploader("CSV 파일 선택 (cp949 인코딩)", type=["csv"])
    if uploaded_file:
        csv_bytes = uploaded_file.read()
elif src_opt == "경로 입력(/mnt 또는 GitHub Raw)":
    path = st.sidebar.text_input("CSV 경로(URL 또는 로컬 경로)", value=default_path)
    if path:
        # URL일 수도 있고 로컬일 수도 있음
        if re.match(r"^https?://", path):
            import requests
            r = requests.get(path)
            r.raise_for_status()
            csv_bytes = r.content
        else:
            with open(path, "rb") as f:
                csv_bytes = f.read()

@st.cache_data(show_spinner=True)
def load_data(csv_bytes: bytes) -> pd.DataFrame:
    # 서울열린데이터광장 CSV는 cp949가 기본. 실패 시 utf-8로 재시도.
    buf = io.BytesIO(csv_bytes)
    try:
        df = pd.read_csv(buf, encoding="cp949")
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, encoding="utf-8")
    # 필요한 타입 정리
    # 사용월: 정수/문자 → 202501 같은 형태
    if "사용월" in df.columns:
        df["사용월"] = df["사용월"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
        # 결측 혹은 비정상 제거
        df = df[df["사용월"].str.len() == 6]
        df["사용월"] = df["사용월"].astype(int)
    # 열 이름 정리 (공백 제거)
    df.columns = [c.strip() for c in df.columns]
    return df

if csv_bytes is None:
    st.info("좌측에서 CSV를 선택/입력하세요. (기본 경로가 보이면 그대로 사용 가능)")
    st.stop()

with st.spinner("CSV 로딩 중..."):
    df_raw = load_data(csv_bytes)

# ---------------------------
# 🧹 전처리: Wide → Long
# ---------------------------
@st.cache_data(show_spinner=True)
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    # 시간대·승/하차 열 탐지
    hour_cols = [c for c in df.columns if ("승차인원" in c or "하차인원" in c)]
    id_cols = [c for c in ["사용월", "호선명", "지하철역", "요일", "작업일자"] if c in df.columns]

    # 시간대 문자열 집합 (예: "07시-08시")
    time_bins = sorted(set([c.split()[0] for c in hour_cols]),
                       key=lambda s: int(re.match(r"(\d+)", s).group(1)))

    # Long 변환: 각 시간대에 대해 승/하차를 한 번에 붙이기
    parts = []
    for t in time_bins:
        bcol = f"{t} 승차인원"
        acol = f"{t} 하차인원"
        use_cols = [c for c in id_cols] + [bcol] + [acol]
        sub = df[use_cols].copy()
        sub["시간대"] = t
        sub.rename(columns={bcol: "승차", acol: "하차"}, inplace=True)
        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True)
    # 시간대 시작 시(hour_start) 추출 (예: "07시-08시" → 7)
    long_df["hour_start"] = long_df["시간대"].str.extract(r"^(\d+)")
    long_df["hour_start"] = pd.to_numeric(long_df["hour_start"], errors="coerce").fillna(0).astype(int)
    # 다운캐스팅
    for col in ["승차", "하차"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce").fillna(0).astype(np.int32)
    if "사용월" in long_df.columns:
        long_df["사용월"] = pd.to_numeric(long_df["사용월"], errors="coerce").fillna(0).astype(np.int32)
    return long_df

with st.spinner("전처리 중 (Wide → Long)…"):
    df = to_long(df_raw)

# 메타 정보
st.success(f"데이터 로드 완료: {len(df):,} 행 | 컬럼: {list(df.columns)}")
with st.expander("데이터 예시 보기", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------
# 🔧 학습/예측 유틸
# ---------------------------
def pick_model(name: str):
    if name == "LightGBM (자동 권장)" and LGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    elif name.startswith("RandomForest"):
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )
    else:
        return LinearRegression()

def safe_label_encode(series: pd.Series):
    le = LabelEncoder()
    vals = series.fillna("N/A").astype(str)
    return le.fit_transform(vals), le

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@st.cache_resource(show_spinner=True)
def train_models(df_long: pd.DataFrame, algo_name: str):
    # 특징: 호선명, 지하철역, hour_start, 사용월, (가능하면 요일)
    feat_cols = []
    for c in ["호선명", "지하철역", "hour_start", "사용월"]:
        if c in df_long.columns:
            feat_cols.append(c)
    if "요일" in df_long.columns:
        feat_cols.append("요일")

    work = df_long[feat_cols + ["승차", "하차"]].dropna().copy()

    # 인코딩
    encoders = {}
    X = pd.DataFrame(index=work.index)
    for c in feat_cols:
        if work[c].dtype == "O":
            X[c], enc = safe_label_encode(work[c])
            encoders[c] = enc
        else:
            X[c] = work[c].astype(np.int32)

    # 타겟(승차, 하차) 각각 모델
    models = {}
    metrics = {}

    X_train, X_test, yb_train, yb_test = train_test_split(X, work["승차"], test_size=0.2, random_state=42)
    model_b = pick_model(algo_name)
    model_b.fit(X_train, yb_train)
    yb_pred = model_b.predict(X_test)
    metrics["승차_RMSE"] = rmse(yb_test, yb_pred)
    metrics["승차_MAE"]  = mean_absolute_error(yb_test, yb_pred)
    metrics["승차_R2"]   = r2_score(yb_test, yb_pred)
    models["승차"] = model_b

    X_train, X_test, ya_train, ya_test = train_test_split(X, work["하차"], test_size=0.2, random_state=42)
    model_a = pick_model(algo_name)
    model_a.fit(X_train, ya_train)
    ya_pred = model_a.predict(X_test)
    metrics["하차_RMSE"] = rmse(ya_test, ya_pred)
    metrics["하차_MAE"]  = mean_absolute_error(ya_test, ya_pred)
    metrics["하차_R2"]   = r2_score(ya_test, ya_pred)
    models["하차"] = model_a

    return {
        "feat_cols": feat_cols,
        "encoders": encoders,
        "models": models,
        "metrics": metrics
    }

# ---------------------------
# 🎛️ 사이드바: 모델 선택/학습
# ---------------------------
st.sidebar.header("2) 모델 학습 설정")
algo_name = st.sidebar.selectbox(
    "알고리즘 선택",
    ["LightGBM (자동 권장)"] + ["RandomForest (대체)", "LinearRegression (간단)"],
    index=0 if LGBM_AVAILABLE else 1
)
with st.sidebar:
    st.caption("💡 LightGBM 설치 안되어 있으면 자동으로 대체 모델 사용")

with st.spinner("모델 학습/검증 중…(캐시됨)"):
    pack = train_models(df, algo_name)
st.sidebar.success("학습 완료!")
with st.sidebar.expander("평가지표 (검증셋)", expanded=False):
    st.json(pack["metrics"])

# ---------------------------
# 🗺️ 입력 위젯
# ---------------------------
st.header("🧮 예측 입력")
cols = st.columns([1, 1, 1, 1.2, 1])
lines = sorted(df["호선명"].dropna().unique().tolist())
sel_line = cols[0].selectbox("호선", lines)

stations = sorted(df.loc[df["호선명"] == sel_line, "지하철역"].dropna().unique().tolist())
sel_station = cols[1].selectbox("역", stations)

# 날짜 → 월(YYYYMM)로 변환하여 사용
sel_date = cols[2].date_input("날짜 선택", value=dt.date.today())
sel_month = int(sel_date.strftime("%Y%m"))
cols[2].caption(f"사용월로 변환: **{sel_month}**")

# 시간대
time_bins = df["시간대"].dropna().unique().tolist()
# 시간대 자연 정렬
def hour_key(s): 
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else 0
time_bins = sorted(time_bins, key=hour_key)
sel_time = cols[3].selectbox("시간대", time_bins)

# 요일 (데이터에 있을 때만 실제로 사용)
weekday_options = ["월","화","수","목","금","토","일"]
sel_weekday = cols[4].selectbox("요일(선택)", options=["(미사용)"]+weekday_options, index=0,
                               help="데이터에 '요일' 컬럼이 존재할 때만 모델에 반영됩니다.")

# ---------------------------
# 🔮 예측
# ---------------------------
def build_feature_row(pack, line, station, month, time_str, weekday):
    feat_cols = pack["feat_cols"]
    hour_start = int(re.match(r"(\d+)", time_str).group(1))
    row = {}
    for c in feat_cols:
        if c == "호선명":
            row[c] = line
        elif c == "지하철역":
            row[c] = station
        elif c == "hour_start":
            row[c] = hour_start
        elif c == "사용월":
            row[c] = month
        elif c == "요일":
            row[c] = weekday if weekday in weekday_options else "N/A"
    # 인코딩
    X = {}
    for c in feat_cols:
        if c in pack["encoders"]:
            enc = pack["encoders"][c]
            X[c] = enc.transform([str(row.get(c, 'N/A'))])[0]
        else:
            X[c] = int(row.get(c, 0))
    return pd.DataFrame([X], columns=feat_cols)

colL, colR = st.columns([1,1])
with colL:
    if st.button("🚀 예측 실행", use_container_width=True):
        X_row = build_feature_row(
            pack, sel_line, sel_station, sel_month, sel_time, sel_weekday
        )
        pred_board = float(pack["models"]["승차"].predict(X_row)[0])
        pred_alight = float(pack["models"]["하차"].predict(X_row)[0])

        st.subheader("📌 예측 결과")
        c1, c2 = st.columns(2)
        c1.metric("예상 **승차** 인원", f"{int(round(pred_board)):,} 명")
        c2.metric("예상 **하차** 인원", f"{int(round(pred_alight)):,} 명")

        # ---------------------------
        # 📊 비교: 과거 분포/평균
        # ---------------------------
        st.markdown("---")
        st.subheader("📈 과거 분포와 비교")
        hist = df[(df["호선명"] == sel_line) & (df["지하철역"] == sel_station) & (df["시간대"] == sel_time)]
        if len(hist) > 0:
            # 월별 통계
            grp = hist.groupby("사용월", as_index=False)[["승차", "하차"]].agg(["mean", "median", "min", "max"])
            grp.columns = [f"{a}_{b}" for a,b in grp.columns]
            grp = grp.reset_index().rename(columns={"index":"사용월"})
            st.dataframe(grp.sort_values("사용월", ascending=False), use_container_width=True)

            # Altair 시각화 (Streamlit 기본 지원)
            import altair as alt
            line1 = alt.Chart(hist).mark_line(point=True).encode(
                x=alt.X("사용월:O", title="사용월(YYYYMM)"),
                y=alt.Y("승차:Q", title="승차 인원"),
                tooltip=["사용월","승차","하차"]
            ).properties(height=280)
            line2 = alt.Chart(hist).mark_line(point=True).encode(
                x=alt.X("사용월:O", title="사용월(YYYYMM)"),
                y=alt.Y("하차:Q", title="하차 인원"),
                tooltip=["사용월","승차","하차"]
            ).properties(height=280)
            st.altair_chart(line1, use_container_width=True)
            st.altair_chart(line2, use_container_width=True)

            # 예측치 vs 최근 월 평균 비교
            recent_mean = hist[hist["사용월"] >= (sel_month - 100)][["승차","하차"]].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("최근 1년 평균 승차", f"{int(round(recent_mean['승차'])):,} 명" if not np.isnan(recent_mean["승차"]) else "데이터 없음")
            c2.metric("최근 1년 평균 하차", f"{int(round(recent_mean['하차'])):,} 명" if not np.isnan(recent_mean["하차"]) else "데이터 없음")
            diff_b = (pred_board - (recent_mean["승차"] if not np.isnan(recent_mean["승차"]) else 0))
            diff_a = (pred_alight - (recent_mean["하차"] if not np.isnan(recent_mean["하차"]) else 0))
            c3.write(f"예측치 대비 최근 1년 평균 차이 — 승차: **{diff_b:+.0f}명**, 하차: **{diff_a:+.0f}명**")
        else:
            st.info("선택한 조합(호선/역/시간대)에 대한 과거 데이터가 없습니다.")

with colR:
    # 간단 EDA: 선택한 호선의 시간대별 평균 히트맵
    st.subheader("🧭 호선 시간대별 평균 (간단 EDA)")
    sub = df[df["호선명"] == sel_line]
    if len(sub) > 0:
        pivot_b = sub.pivot_table(index="지하철역", columns="hour_start", values="승차", aggfunc="mean")
        pivot_a = sub.pivot_table(index="지하철역", columns="hour_start", values="하차", aggfunc="mean")
        st.caption("승차 평균 히트맵")
        st.dataframe(pivot_b.fillna(0).astype(int), use_container_width=True, height=300)
        st.caption("하차 평균 히트맵")
        st.dataframe(pivot_a.fillna(0).astype(int), use_container_width=True, height=300)
    else:
        st.info("해당 호선 데이터가 없습니다.")

st.markdown("---")
with st.expander("ℹ️ 주의/설명", expanded=False):
    st.markdown("""
- 이 데이터셋은 **월 단위(`사용월`)**와 **시간대(예: `07시-08시`)** 집계입니다.  
- 공개 CSV에 **`요일` 컬럼이 없으면** 요일 입력은 **예측에 반영되지 않습니다.** (UI에서만 선택 가능)  
- LightGBM 미설치 환경에서는 자동으로 **RandomForest → LinearRegression** 으로 대체합니다.  
- 예측값은 과거 패턴 기반 통계적 추정이므로 실제 탑승 수요와 차이가 날 수 있습니다.
""")
