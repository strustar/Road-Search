#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE-m3 RAG 검색 + LLM 답변 시스템 (UI/최적화 업그레이드)
- 최신 LLM 선택 지원(드롭다운 + 커스텀 모델)
- 참조 문서 수 AUTO 최적화(기본)
- 참조 문서 상세: 모두 펼침 + 가독성 강화 + 표 전체 표시
- 문서 기반 답변 엄격화(시스템 프롬프트 강화)
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st

try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_AVAILABLE = True
except:
    BGE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="설계실무지침 AI 검색",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 스타일 (가독성 업그레이드)
# ------------------------------
st.markdown("""
<style>
  .main { background-color: #f8f9fa; }
  .stButton > button { width: 100%; }
  .answer-box {
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .result-card {
    background: #ffffff;
    padding: 1rem 1.1rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    margin: 0.6rem 0;
  }
  .result-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.4rem;
  }
  .pill {
    display:inline-block; padding: 0.12rem 0.5rem; border-radius: 999px;
    border:1px solid #dee2e6; background:#f8f9fa; font-size: 0.8rem; color:#495057;
    margin-right: 0.4rem;
  }
  .meta {
    font-size: 0.85rem; color: #6c757d; margin-top: 0.15rem;
  }
  .codebox {
    background:#fcfcfd; border:1px solid #eee; border-radius:10px; padding:0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 0.9rem; white-space: pre-wrap;
  }
  /* 데이터프레임 너비 확보 */
  .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# 임베딩 매니저 (간소화)
# =========================
class EmbeddingManager:
    def __init__(self):
        self.model: Optional[BGEM3FlagModel] = None
        self.client: Optional[OpenAI] = None
        self.embeddings_dense: Optional[np.ndarray] = None
        self.embeddings_dense_norm: Optional[np.ndarray] = None
        self.embeddings_sparse: Optional[Any] = None
        self.chunks: Optional[List[Dict[str, Any]]] = None
        self.embedding_dim: Optional[int] = None
        self.device = "cpu"
        self._qcache: Dict[str, Any] = {}

    @st.cache_resource(show_spinner=False)
    def load_bge_model(_self):
        if not BGE_AVAILABLE:
            return None
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except:
            pass
        with st.spinner(f"🔄 BGE-m3 모델 로딩... (device={device})"):
            model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(device=="cuda"))
        _self.device = device
        return model

    def init_openai(self):
        if not OPENAI_AVAILABLE:
            st.error("❌ openai 패키지가 설치되지 않았습니다.")
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ .env 파일에 OPENAI_API_KEY가 없습니다!")
            return None
        try:
            self.client = OpenAI(api_key=api_key)
            return self.client
        except Exception as e:
            st.error(f"❌ OpenAI 클라이언트 초기화 실패: {str(e)}")
            return None

    def load_manifest(self, base_dir: Path) -> Dict[str, Any]:
        mp = base_dir / "manifest.json"
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"shards": {}}

    def load_shard_npz(self, base_dir: Path, shard_name: str):
        try:
            mani = self.load_manifest(base_dir)
            if shard_name not in mani.get("shards", {}):
                raise RuntimeError(f"샤드 '{shard_name}' 를 찾을 수 없습니다.")
            meta = mani["shards"][shard_name]

            # Dense
            npz_file = base_dir / meta["dense"]["file"]
            with np.load(npz_file, allow_pickle=True) as npz:
                self.embeddings_dense = np.array(npz["dense"])
                self.embeddings_dense_norm = np.array(npz["dense_norm"])
                self.embedding_dim = int(npz["dim"][0])
                self.device = str(npz["device"][0])

            # Sparse
            sparse_meta = meta["sparse"]
            if sparse_meta["type"] == "csr" and SCIPY_AVAILABLE:
                csr_file = base_dir / sparse_meta["file"]
                self.embeddings_sparse = sp.load_npz(csr_file).tocsr().astype(np.float32)
            else:
                with open(base_dir / sparse_meta["file"], "r", encoding="utf-8") as f:
                    js = json.load(f)
                self.embeddings_sparse = {int(k): {int(kk): float(vv) for kk, vv in d.items()}
                                          for k, d in js.items()}

            # Chunks
            chunks_file = base_dir / meta["chunks"]
            self.chunks = []
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))

            self._qcache.clear()

        except Exception as e:
            st.error(f"❌ 샤드 로드 실패: {str(e)}")
            raise

    def encode_query(self, query: str, max_len: int = 512):
        key = f"{query}|{max_len}"
        if key in self._qcache:
            return self._qcache[key]["dense"], self._qcache[key]["sparse"]

        if not self.model:
            self.model = self.load_bge_model()
        if not self.model:
            st.error("❌ BGE-m3 모델을 로드하지 못했습니다.")
            raise RuntimeError("BGE model unavailable")

        out = self.model.encode(
            [query], batch_size=1, max_length=max_len,
            return_dense=True, return_sparse=True, return_colbert_vecs=False
        )

        q_dense = out["dense_vecs"][0].astype(np.float32)
        q_dense /= (np.linalg.norm(q_dense) + 1e-12)

        q_sparse = out.get("lexical_weights", [{}])[0]
        q_sparse = {int(k): float(v) for k, v in q_sparse.items()}

        self._qcache[key] = {"dense": q_dense, "sparse": q_sparse}
        return q_dense, q_sparse

    def _dense_scores(self, q_dense: np.ndarray) -> np.ndarray:
        return self.embeddings_dense_norm @ q_dense

    def _sparse_scores(self, q_sparse: Dict[int, float]) -> np.ndarray:
        n = len(self.chunks)
        if not q_sparse:
            return np.zeros(n, dtype=np.float32)

        if isinstance(self.embeddings_sparse, sp.csr_matrix):
            q_vec = np.zeros(self.embeddings_sparse.shape[1], dtype=np.float32)
            for tid, weight in q_sparse.items():
                if tid < len(q_vec):
                    q_vec[tid] = weight
            return np.array(self.embeddings_sparse @ q_vec).flatten()

        scores = np.zeros(n, dtype=np.float32)
        for idx, doc_sparse in self.embeddings_sparse.items():
            s = 0.0
            for tid, qw in q_sparse.items():
                dv = doc_sparse.get(tid)
                if dv is not None:
                    s += qw * dv
            scores[idx] = s
        return scores

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid",
               dense_weight: float = 0.6) -> List[Dict[str, Any]]:
        if self.embeddings_dense_norm is None or self.chunks is None:
            return []

        q_dense, q_sparse = self.encode_query(query)

        if mode == "dense":
            scores = self._dense_scores(q_dense)
        elif mode == "sparse":
            scores = self._sparse_scores(q_sparse)
        else:  # hybrid
            dscore = self._dense_scores(q_dense)
            sscore = self._sparse_scores(q_sparse)
            # 간단 표준화 후 가중합
            dscore = (dscore - dscore.mean()) / (dscore.std() + 1e-12)
            sscore = (sscore - sscore.mean()) / (sscore.std() + 1e-12)
            scores = dense_weight * dscore + (1 - dense_weight) * sscore

        # 표 부스팅(질문에 표/수치 관련 키워드가 있으면 표 포함 청크 가중)
        table_terms = ["표", "구분", "연락처", "전화", "부서", "노선", "kN", "MPa", "mm", "기준", "수치"]
        if any(t in query for t in table_terms):
            boost = np.array([1.05 if c.get("metadata", {}).get("has_table") else 1.0
                              for c in self.chunks], dtype=np.float32)
            scores = scores * boost

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            results.append({
                "rank": rank,
                "index": int(idx),
                "score": float(scores[idx]),
                "chunk": self.chunks[idx]
            })
        return results

# =========================
# 유틸: 컨텍스트/토큰/오토-K
# =========================
def approx_tokens(text: str) -> int:
    # 대략 1토큰 ≈ 4자(한글/영문 혼재 기준 대충치)
    return max(1, len(text) // 4)

def build_context_from_results(results: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    context_parts = []
    for i, result in enumerate(results[:max_chunks], 1):
        chunk = result["chunk"]

        heading_parts = []
        if chunk.get("big_heading"):
            heading_parts.append(str(chunk["big_heading"]))
        if chunk.get("mid_heading") and chunk["mid_heading"] != chunk.get("big_heading"):
            heading_parts.append(str(chunk["mid_heading"]))
        heading = " > ".join(heading_parts) if heading_parts else "제목 없음"

        items = chunk.get("items", []) or []
        text_parts = []

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                txt = (item.get("content") or "").strip()
                if txt:
                    text_parts.append(txt)
            elif item.get("type") == "table":
                title = item.get("title", "표")
                tbl = item.get("table_data", {})
                cols = [str(c) for c in tbl.get("columns", [])]
                rows = tbl.get("data", [])
                table_str = f"\n[{title}]\n"
                if cols:
                    table_str += " | ".join(cols) + "\n"
                for row in rows:
                    table_str += " | ".join([str(x) if x is not None else "" for x in row]) + "\n"
                text_parts.append(table_str)

        content = "\n".join(text_parts) if text_parts else (chunk.get("content", "")[:2000])

        meta = chunk.get("metadata", {})
        source = meta.get("source_file", meta.get("file_name", "unknown"))

        context_parts.append(
            f"[문서 {i}]\n"
            f"출처: {source}\n"
            f"제목: {heading}\n"
            f"내용:\n{content}\n"
            f"{'='*60}"
        )
    return "\n\n".join(context_parts)

def choose_auto_context_k(query: str, results: List[Dict[str, Any]], max_cap: int = 10) -> int:
    """
    간단 휴리스틱:
    - 기본 6개
    - 질문이 길수록 +1 (>=80자)
    - 표/수치 키워드 포함 시 +1
    - 결과가 매우 적으면 len(results)로 제한
    - 상한 10
    """
    k = 6
    if len(query) >= 80:
        k += 1
    if any(t in query for t in ["표", "기준", "수치", "절차", "단계", "MPa", "kN", "mm"]):
        k += 1
    k = min(k, len(results), max_cap)
    return max(1, k)

# =========================
# LLM 함수들
# =========================
STRICT_SYSTEM_PROMPT = """
당신은 한국도로공사 '설계실무지침' 전문가다. 반드시 제공된 문서 내용만 사용하여 답하라.
문서에 없는 내용은 추정/발명/일반상식 보충을 금지한다.
찾을 수 없으면 정확히 다음 문구로 답한다: "제공된 문서에서 해당 정보를 찾을 수 없습니다."

[출력 형식(웹 가독성 최적화)]
- Markdown을 사용한다.
- 상단에 요약(3~5줄) → 그 아래에 세부내용을 번호 목록으로 단계/절차/조건을 정리한다.
- 수치·기준은 단위와 범위를 반드시 함께 적는다(예: 1.5 m, 30 MPa, ±5%).
- 표가 필요하면:
  1) 문서 표의 핵심 행/열만 뽑아 **Markdown 표**로 재구성하거나,
  2) 원본 표가 그대로 전달되어야 의미가 보존될 때는 **"원본 표 그대로"**를 표시하고 표 내용을 가능한 한 그대로 제시한다.
- 긴 항목은 하위 불릿(-)을 활용해 줄바꿈과 들여쓰기로 가독성을 높인다.
- 마지막에는 **참조** 섹션을 넣되, [문서 1] 같은 번호 표기는 금지하고,
  각 근거별로 **출처 파일명 · 큰제목 · 중간제목**을 ‘·’로 구분해 나열한다.
  (예시: 설계실무지침_2019.pdf · 3. 도로구조물 설계기준 · 3.2 하중조합)

[내용 작성 규칙]
1) 질문과 직접 관련된 조항, 절차, 책임주체, 시기, 예외조건을 **우선순위**로 정리한다.
2) 서로 상충하는 기술 기준이 있으면 **근거별로 분리**하여 병기하고, 적용 조건을 명시한다.
3) 표/목록에 포함되는 수치는 **원문 수치 그대로** 사용하며, 단위를 누락하지 않는다.
4) 수식/기호가 등장하면 간단히 풀어쓴 설명을 함께 붙인다.
5) 답변 본문 중 불가피한 인용은 짧게 요약 인용만 하고(두세 줄 이내), 원문 전체 제시는 피한다.
6) 제공 컨텍스트 외 배경지식, 타 규정/코드, 일반 상식, 개인적 판단은 사용 금지.

[참조 표기 방법(필수)]
- 본문 마지막에 다음 형식으로 나열:
  - 출처 파일명 · 큰제목 · 중간제목
  - (여러 개면 줄바꿈으로 각각 별도 표기)
"""


def generate_llm_response_streaming(client: OpenAI, query: str, context: str,
                                    model: str, placeholder) -> str:
    user_prompt = f"""질문: {query}

참조 문서:
{context}

위 문서에서 근거를 찾아 질문에만 답해라. 문서에 없는 내용은 답하지 말라.
"""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_completion_tokens=3000,
            stream=True
        )
        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                placeholder.markdown(response_text + "▌")
        placeholder.markdown(response_text)
        return response_text
    except Exception as e:
        placeholder.error(f"❌ LLM 호출 오류: {str(e)}")
        return ""

# =========================
# UI: 참조 문서 상세(모두 펼침)
# =========================
def render_chunk_full(result: Dict[str, Any]):
    idx = result["index"]
    score = result["score"]
    chunk = result["chunk"]

    heading_parts = []
    if chunk.get("big_heading"):
        heading_parts.append(str(chunk["big_heading"]))
    if chunk.get("mid_heading") and chunk["mid_heading"] != chunk.get("big_heading"):
        heading_parts.append(str(chunk["mid_heading"]))
    heading = " > ".join(heading_parts) if heading_parts else "제목 없음"

    meta = chunk.get("metadata", {})
    source_folder = meta.get("source_folder", "N/A")
    source_file = meta.get("source_file", meta.get("file_name", "N/A"))
    page = meta.get("page", None)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-header"><div>'
        f'<span class="pill">청크 #{idx}</span>'
        f'<span class="pill">점수 {score:.4f}</span>'
        f'</div><div style="font-weight:600;">📌 {heading}</div></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="meta">📁 {source_folder} &nbsp;&middot;&nbsp; '
        f'📄 {source_file}'
        + (f' &nbsp;&middot;&nbsp; 📃 p.{page}' if page is not None else '')
        + '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    items = chunk.get("items", []) or []
    # 모든 내용 표시(표는 전체, 텍스트는 모두)
    for j, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            txt = (item.get("content") or "").strip()
            if txt:
                st.markdown(txt)
        elif item.get("type") == "table":
            title = item.get("title", "표")
            tbl = item.get("table_data", {})
            df = pd.DataFrame(tbl.get("data", []))
            cols = [str(c) for c in tbl.get("columns", [])]
            if cols and len(cols) == df.shape[1]:
                df.columns = cols
            st.markdown(f"**📊 {title}**")
            st.dataframe(df, use_container_width=True)
        elif item.get("type") == "code":
            code_text = item.get("content", "")
            st.markdown(f"**🔧 코드**")
            st.markdown(f"<div class='codebox'>{code_text}</div>", unsafe_allow_html=True)
        else:
            # 알 수 없는 타입은 문자열화
            st.json(item)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 메인
# =========================
def main():
    st.title("🏗️ 설계실무지침 AI 검색")
    st.caption("BGE-m3 임베딩 + GPT 답변 생성")

    # 초기화
    if "manager" not in st.session_state:
        st.session_state.manager = EmbeddingManager()
    manager: EmbeddingManager = st.session_state.manager

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")

        # OpenAI
        if st.button("🔑 OpenAI 연결 확인"):
            if manager.init_openai():
                st.success("✅ 연결 성공!")

        st.divider()

        # 샤드 로드
        st.subheader("📂 임베딩 로드")
        base_dir = Path(st.text_input("샤드 폴더", value=".embeddings_shards"))
        if not base_dir.exists():
            st.error("❌ 폴더가 존재하지 않습니다!")
            st.stop()

        mani = manager.load_manifest(base_dir)
        shard_names = list(mani.get("shards", {}).keys())
        if not shard_names:
            st.warning("⚠️ 샤드가 없습니다.")
            st.stop()

        selected_shard = st.selectbox("샤드 선택", shard_names)
        if st.button("🔄 샤드 로드", type="primary"):
            with st.spinner("로딩 중..."):
                manager.load_shard_npz(base_dir, selected_shard)
            st.success(f"✅ {len(manager.chunks):,}개 청크 로드!")

        st.divider()

        # LLM 설정 (최신 선택 + 커스텀)
        st.subheader("🤖 LLM 설정")
        # 드롭다운 후보는 예시이며, 계정에서 지원되는 최신 모델명을 커스텀 입력으로 추가 사용 가능
        model_options = {
            "GPT-5 (2025년 최신)": "gpt-5",
            "GPT-5-mini (경량 GPT-5)": "gpt-5-mini",
            "GPT-5-nano (초경량)": "gpt-5-nano",
            "o1 (추론 모델)": "o1",
            "o1-preview": "o1-preview",
            "o1-mini": "o1-mini",
            "GPT-4o (안정)": "gpt-4o",
            "GPT-4o-mini (추천/저비용)": "gpt-4o-mini",  # ⭐ 기본값
            "GPT-4-turbo": "gpt-4-turbo"
        }
        model_display = st.selectbox("모델", list(model_options.keys()), index=6)
        selected_model = model_options[model_display]
        custom_model = st.text_input("추가/커스텀 모델명(선택)", value="", placeholder="예: gpt-5.1, gpt-5o-reasoning 등")
        if custom_model.strip():
            selected_model = custom_model.strip()

        st.divider()

        # 검색 설정
        st.subheader("🔍 검색 설정")
        search_mode = st.radio("모드", ["Hybrid", "Dense", "Sparse"], index=0, horizontal=True)
        mode_map = {"Hybrid": "hybrid", "Dense": "dense", "Sparse": "sparse"}
        mode = mode_map[search_mode]
        dense_weight = st.slider("Dense 가중치", 0.0, 1.0, 0.6, 0.1) if mode == "hybrid" else 0.6
        top_k = st.slider("검색 결과 수(Top-K)", 3, 30, 12, 1)

        # 참조 문서 수: AUTO 기본
        st.subheader("🧠 참조 문서 수")
        use_auto = st.checkbox("AUTO(권장)", value=True,
                               help="질문/키워드에 따라 4~10개 사이로 자동 선택")
        context_chunks_manual = st.slider("수동 설정(비활성 시 무시)", 1, 15, 6, 1)

        st.divider()

        if manager.chunks:
            st.metric("📊 청크", f"{len(manager.chunks):,}")
            st.metric("🔢 차원", manager.embedding_dim)

    # 메인
    if manager.chunks is None:
        st.info("👈 사이드바에서 샤드를 먼저 로드하세요!")
        st.stop()

    if manager.client is None:
        manager.init_openai()
        if manager.client is None:
            st.warning("⚠️ OpenAI API 키를 설정하세요!")
            st.stop()

    # 예제 질문
    st.subheader("📝 예제 질문")
    example_questions = [
        "설계 안전성 검토가 무엇이고 어떻게 적용하는지 알려줘",
        "설계단계별 주민설명회를 언제 해야하는지 알려줘",
        "지적중첩도 작성 의뢰를 언제 해야하는지 알려줘",
        "확장구간 내 제한속도와 최소 곡선반경을 알려줘",
        "점검 승강시설이 무엇이고 어디에 설치해야 하는지 알려줘",
        "현광방지시설 설치방법을 알려줘",
        "통과높이 제한시설 설치기준 알려줘",
        "품질시험사 인건비 반영 방법을 알려줘"
    ]
    cols = st.columns(4)
    for i, q in enumerate(example_questions):
        with cols[i % 4]:
            if st.button(f"Q{i+1}", key=f"ex_{i}", use_container_width=True, help=q):
                st.session_state.selected_query = q
                st.rerun()

    st.divider()

    # 검색창
    default_query = st.session_state.get("selected_query", "")
    query = st.text_input("🔍 질문을 입력하세요", value=default_query,
                          placeholder="예: 설계 안전성 검토란?")
    if "selected_query" in st.session_state:
        del st.session_state.selected_query

    # 실행
    if st.button("🚀 검색 & 답변 생성", type="primary", use_container_width=True) or default_query:
        if not query:
            st.warning("⚠️ 질문을 입력하세요!")
            st.stop()

        # 1) 검색
        with st.spinner("🔍 문서 검색 중..."):
            t0 = time.time()
            results = manager.search(query, top_k=top_k, mode=mode, dense_weight=dense_weight)
            search_time = time.time() - t0

        if not results:
            st.warning("⚠️ 관련 문서를 찾을 수 없습니다.")
            st.stop()

        st.success(f"✅ {len(results)}개 문서 검색 완료 ({search_time:.2f}초)")

        # 2) 컨텍스트 구성 (AUTO 기본)
        if use_auto:
            k_auto = choose_auto_context_k(query, results, max_cap=10)
            context = build_context_from_results(results, max_chunks=k_auto)
            chosen_k = k_auto
        else:
            context = build_context_from_results(results, max_chunks=context_chunks_manual)
            chosen_k = context_chunks_manual

        # 3) LLM 답변
        st.markdown("### 🤖 AI 답변")
        answer_placeholder = st.empty()
        with st.spinner(f"💭 {selected_model} 답변 생성 중..."):
            _ = generate_llm_response_streaming(
                manager.client, query, context, selected_model, answer_placeholder
            )

        # 4) 참조 문서 상세(모두 펼침 + 가독성 강화)
        st.markdown("---")
        st.markdown("### 📚 참조 문서 상세")
        for r in results:
            render_chunk_full(r)

        # 통계
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ 검색 시간", f"{search_time:.2f}초")
        with col2:
            st.metric("📄 검색 결과", f"{len(results)}개")
        with col3:
            st.metric("🧠 LLM 모델", selected_model)
        with col4:
            st.metric("📚 LLM 컨텍스트 문서", f"{chosen_k}개")

if __name__ == "__main__":
    if not BGE_AVAILABLE:
        st.error("❌ FlagEmbedding 미설치")
        st.code("pip install FlagEmbedding")
        st.stop()
    if not OPENAI_AVAILABLE:
        st.error("❌ openai 패키지 미설치")
        st.code("pip install openai python-dotenv")
        st.stop()
    main()
