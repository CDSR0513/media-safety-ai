import io, json, datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from detection.detector import read_image, run_detection_bgr, draw_boxes, summarize

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_CSV = DATA_DIR / "results.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
if not RESULTS_CSV.exists():
    RESULTS_CSV.write_text("date,filename,platform,total,classes_json\n", encoding="utf-8")

st.set_page_config(page_title="DrinkVision MVP", layout="wide")

st.sidebar.title("DrinkVision MVP")
mode = st.sidebar.radio("메뉴", ["업로드/분석", "통계"])
platform = st.sidebar.selectbox("플랫폼 태그", ["YouTube", "OTT", "Instagram", "Etc"])
today = dt.date.today().isoformat()

def append_result(row):
    with RESULTS_CSV.open("a", encoding="utf-8") as f:
        f.write(",".join([
            row["date"],
            row["filename"],
            row["platform"],
            str(row["total"]),
            json.dumps(row["classes_json"], ensure_ascii=False)
        ]) + "\n")

if mode == "업로드/분석":
    st.header("업로드해서 바로 감지")
    files = st.file_uploader("이미지 여러 장 / 짧은 영상은 이미지로 먼저", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if files:
        cols = st.columns(2)
        for file in files:
            # 원본 저장
            raw_path = RAW_DIR / file.name
            raw_path.write_bytes(file.getvalue())

            # 감지
            bgr = read_image(io.BytesIO(file.getvalue()))
            result = run_detection_bgr(bgr)
            total, cls_map = summarize(result)
            out_bgr = draw_boxes(bgr, result)
            out_path = PROC_DIR / f"det_{file.name}"
            import cv2
            cv2.imwrite(str(out_path), out_bgr)

            # 화면 표시
            with cols[0]:
                st.subheader(file.name)
                st.image(bgr[..., ::-1], caption="원본", use_container_width=True)
            with cols[1]:
                st.image(out_bgr[..., ::-1], caption=f"감지 결과 (총 {total})", use_container_width=True)
                st.json(cls_map)

            # CSV 누적
            append_result({
                "date": today,
                "filename": file.name,
                "platform": platform,
                "total": total,
                "classes_json": cls_map
            })

        st.success("처리 완료! 통계 탭에서 누적 결과를 확인하세요.")

elif mode == "통계":
    st.header("누적 통계")
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        st.info("아직 결과가 없습니다. 먼저 업로드/분석을 해보세요.")
    else:
        df = pd.read_csv(RESULTS_CSV)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("일자별 총 탐지수")
            g = df.groupby("date")["total"].sum().reset_index()
            st.plotly_chart(px.bar(g, x="date", y="total"), use_container_width=True)
        with c2:
            st.subheader("플랫폼별 총 탐지수")
            g2 = df.groupby("platform")["total"].sum().reset_index()
            st.plotly_chart(px.bar(g2, x="platform", y="total"), use_container_width=True)

        st.subheader("원시 데이터")
        st.dataframe(df, use_container_width=True)
        st.download_button("results.csv 다운로드", data=RESULTS_CSV.read_bytes(), file_name="results.csv")
