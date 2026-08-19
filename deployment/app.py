
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="RxReview",
    page_icon="💊",
    layout="wide"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

EVALUATION_DATA_PATH = (
    "rxreview_priority_data.csv"
)

SCORING_DATA_PATH = (
    "rxreview_scoring_input.csv"
)

MODEL_DIR = os.path.join(
    "models",
    "xgboost"
)

XGB_PIPELINE_PATH = os.path.join(
    MODEL_DIR,
    "rxreview_xgb_pipeline.joblib"
)

CALIBRATOR_PATH = os.path.join(
    MODEL_DIR,
    "rxreview_isotonic_calibrator.joblib"
)

MODEL_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

RAG_DIR = "rag"

RAG_CHUNKS_PATH = os.path.join(
    RAG_DIR,
    "rxreview_rag_chunks.csv"
)

RAG_INDEX_PATH = os.path.join(
    RAG_DIR,
    "rxreview_faiss.index"
)

CAPACITY_OPTIONS = [
    5,
    10,
    15,
    20,
    25,
    30
]

BASELINE_HOURLY_COST = 75.0
BASELINE_REVIEW_MINUTES = 30
BASELINE_READMISSION_COST = 15000.0
BASELINE_PREVENTABLE_FRACTION = 0.25
BASELINE_EFFECTIVENESS = 0.20


# ---------------------------------------------------------
# LOAD EVALUATION DATA
# ---------------------------------------------------------

@st.cache_data
def load_evaluation_data():

    return pd.read_csv(
        EVALUATION_DATA_PATH
    )


priority_df = (
    load_evaluation_data()
)


# ---------------------------------------------------------
# LOAD SAVED XGBOOST MODEL ASSETS
# ---------------------------------------------------------

@st.cache_resource
def load_xgb_assets():

    pipeline = joblib.load(
        XGB_PIPELINE_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    with open(
        MODEL_CONFIG_PATH,
        "r"
    ) as f:

        config = json.load(
            f
        )

    return (
        pipeline,
        calibrator,
        config
    )


@st.cache_data
def load_scoring_input():

    return pd.read_csv(
        SCORING_DATA_PATH
    )


def score_operational_population(
    scoring_df
):

    (
        xgb_pipeline,
        calibrator,
        model_config
    ) = load_xgb_assets()

    numeric_features = (
        model_config[
            "numeric_features"
        ]
    )

    categorical_features = (
        model_config[
            "categorical_features"
        ]
    )

    model_features = (
        numeric_features
        +
        categorical_features
    )

    missing_features = [
        col
        for col in model_features
        if col not in scoring_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Live-scoring data is missing "
            "required model features: "
            + ", ".join(
                missing_features
            )
        )

    result = scoring_df.copy()

    raw_prob = (
        xgb_pipeline.predict_proba(
            result[
                model_features
            ]
        )[:, 1]
    )

    calibrated_prob = (
        calibrator.predict(
            raw_prob
        )
    )

    result[
        "xgb_live_raw_readmission_risk"
    ] = raw_prob

    result[
        "xgb_calibrated_readmission_risk"
    ] = calibrated_prob


    # -----------------------------------------------------
    # Recreate Medication Complexity Score
    # -----------------------------------------------------

    med_cap = max(
        result[
            "num_medications"
        ].quantile(
            0.95
        ),
        1
    )

    active_med_cap = max(
        result[
            "active_diabetes_meds"
        ].quantile(
            0.95
        ),
        1
    )

    change_cap = max(
        result[
            "diabetes_med_changes"
        ].quantile(
            0.95
        ),
        1
    )

    medication_burden_component = (
        result[
            "num_medications"
        ]
        .clip(
            upper=med_cap
        )
        /
        med_cap
    )

    active_diabetes_med_component = (
        result[
            "active_diabetes_meds"
        ]
        .clip(
            upper=active_med_cap
        )
        /
        active_med_cap
    )

    med_change_component = (
        result[
            "diabetes_med_changes"
        ]
        .clip(
            upper=change_cap
        )
        /
        change_cap
    )

    insulin_component = (
        result[
            "insulin_used"
        ].astype(
            float
        )
    )

    insulin_change_component = (
        result[
            "insulin_changed"
        ].astype(
            float
        )
    )

    result[
        "medication_complexity_score"
    ] = (
        0.30
        *
        medication_burden_component
        +
        0.15
        *
        active_diabetes_med_component
        +
        0.25
        *
        med_change_component
        +
        0.15
        *
        insulin_component
        +
        0.15
        *
        insulin_change_component
    )

    result[
        "medication_complexity_weight"
    ] = (
        1
        +
        result[
            "medication_complexity_score"
        ]
    )


    # -----------------------------------------------------
    # Calculate live RxReview Priority Score
    # -----------------------------------------------------

    result[
        "rxreview_priority_score"
    ] = (
        result[
            "xgb_calibrated_readmission_risk"
        ]
        *
        result[
            "medication_complexity_weight"
        ]
    )

    min_score = (
        result[
            "rxreview_priority_score"
        ].min()
    )

    max_score = (
        result[
            "rxreview_priority_score"
        ].max()
    )

    if max_score > min_score:

        result[
            "rxreview_priority_score_100"
        ] = (
            100
            *
            (
                result[
                    "rxreview_priority_score"
                ]
                -
                min_score
            )
            /
            (
                max_score
                -
                min_score
            )
        )

    else:

        result[
            "rxreview_priority_score_100"
        ] = 0.0


    # -----------------------------------------------------
    # Recreate operational priority categories
    # -----------------------------------------------------

    complexity_pct_rank = (
        result[
            "medication_complexity_score"
        ]
        .rank(
            pct=True,
            method="average"
        )
    )

    result[
        "medication_complexity_level"
    ] = np.select(
        [
            complexity_pct_rank > 0.80,
            complexity_pct_rank > 0.50
        ],
        [
            "High",
            "Moderate"
        ],
        default="Standard"
    )

    priority_pct_rank = (
        result[
            "rxreview_priority_score"
        ]
        .rank(
            pct=True,
            method="average"
        )
    )

    result[
        "rxreview_priority_level"
    ] = np.select(
        [
            priority_pct_rank > 0.90,
            priority_pct_rank > 0.70
        ],
        [
            "High",
            "Moderate"
        ],
        default="Standard"
    )

    return result


@st.cache_data
def build_live_priority_data():

    scoring_input = (
        load_scoring_input()
    )

    return (
        score_operational_population(
            scoring_input
        )
    )


live_priority_df = (
    build_live_priority_data()
)


# ---------------------------------------------------------
# SHARED SESSION STATE
# ---------------------------------------------------------

if "review_capacity_pct" not in st.session_state:

    st.session_state[
        "review_capacity_pct"
    ] = 10


def sync_capacity_widget(
    widget_key
):

    st.session_state[
        widget_key
    ] = st.session_state[
        "review_capacity_pct"
    ]


def update_shared_capacity(
    widget_key
):

    st.session_state[
        "review_capacity_pct"
    ] = st.session_state[
        widget_key
    ]


def capacity_selector(
    label,
    widget_key
):

    sync_capacity_widget(
        widget_key
    )

    return st.radio(
        label,
        options=CAPACITY_OPTIONS,
        format_func=
            lambda x: f"{x}%",
        horizontal=True,
        key=widget_key,
        on_change=
            update_shared_capacity,
        args=(
            widget_key,
        )
    )


# ---------------------------------------------------------
# VALUE / CAPACITY FUNCTIONS
# ---------------------------------------------------------

def select_by_capacity(
    data,
    score_col,
    capacity_fraction
):

    temp = data.sort_values(
        score_col,
        ascending=False
    ).reset_index(
        drop=True
    )

    n_selected = int(
        np.ceil(
            len(temp)
            *
            capacity_fraction
        )
    )

    return temp.iloc[
        :n_selected
    ].copy()


def rxreview_value_simulator(
    data,
    capacity_fraction,
    pharmacist_hourly_cost,
    review_minutes_per_patient,
    cost_per_readmission,
    preventable_fraction,
    intervention_effectiveness
):

    selected = select_by_capacity(
        data,
        "rxreview_priority_score",
        capacity_fraction
    )

    patients_selected = len(
        selected
    )

    total_readmissions = int(
        data[
            "readmit_30"
        ].sum()
    )

    captured_readmissions = int(
        selected[
            "readmit_30"
        ].sum()
    )

    pharmacist_hours = (
        patients_selected
        *
        review_minutes_per_patient
        /
        60
    )

    intervention_cost = (
        pharmacist_hours
        *
        pharmacist_hourly_cost
    )

    capture_rate = (
        captured_readmissions
        /
        total_readmissions
        if total_readmissions > 0
        else 0
    )

    lift = (
        capture_rate
        /
        capacity_fraction
        if capacity_fraction > 0
        else 0
    )

    potentially_preventable = (
        captured_readmissions
        *
        preventable_fraction
    )

    avoided_readmissions = (
        potentially_preventable
        *
        intervention_effectiveness
    )

    gross_savings = (
        avoided_readmissions
        *
        cost_per_readmission
    )

    net_value = (
        gross_savings
        -
        intervention_cost
    )

    denominator = (
        captured_readmissions
        *
        preventable_fraction
        *
        cost_per_readmission
    )

    break_even_effectiveness = (
        intervention_cost
        /
        denominator
        if denominator > 0
        else np.nan
    )

    return {

        "patients_selected":
            patients_selected,

        "pharmacist_hours":
            pharmacist_hours,

        "captured_readmissions":
            captured_readmissions,

        "capture_rate":
            capture_rate,

        "lift":
            lift,

        "intervention_cost":
            intervention_cost,

        "avoided_readmissions":
            avoided_readmissions,

        "gross_savings":
            gross_savings,

        "net_value":
            net_value,

        "break_even_effectiveness":
            break_even_effectiveness,

        "selected":
            selected
    }


@st.cache_data
def build_capacity_summary(
    data
):

    total_readmissions = int(
        data[
            "readmit_30"
        ].sum()
    )

    rows = []

    for capacity_pct in (
        CAPACITY_OPTIONS
    ):

        capacity_fraction = (
            capacity_pct
            /
            100
        )

        selected = (
            select_by_capacity(
                data,
                "rxreview_priority_score",
                capacity_fraction
            )
        )

        captured = int(
            selected[
                "readmit_30"
            ].sum()
        )

        capture_rate = (
            captured
            /
            total_readmissions
            if total_readmissions > 0
            else 0
        )

        lift = (
            capture_rate
            /
            capacity_fraction
            if capacity_fraction > 0
            else 0
        )

        sim = (
            rxreview_value_simulator(
                data,
                capacity_fraction,
                BASELINE_HOURLY_COST,
                BASELINE_REVIEW_MINUTES,
                BASELINE_READMISSION_COST,
                BASELINE_PREVENTABLE_FRACTION,
                BASELINE_EFFECTIVENESS
            )
        )

        rows.append({

            "Capacity %":
                capacity_pct,

            "Patients Reviewed":
                len(
                    selected
                ),

            "Readmissions Captured":
                captured,

            "Capture Rate %":
                capture_rate
                *
                100,

            "Lift vs Random":
                lift,

            "Net Value":
                sim[
                    "net_value"
                ]
        })

    return pd.DataFrame(
        rows
    )


capacity_summary_df = (
    build_capacity_summary(
        priority_df
    )
)


# ---------------------------------------------------------
# RAG LOADERS
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return (
        SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


@st.cache_resource
def load_faiss_index():

    return faiss.read_index(
        RAG_INDEX_PATH
    )


@st.cache_data
def load_rag_chunks():

    return pd.read_csv(
        RAG_CHUNKS_PATH
    )


@st.cache_resource
def load_generation_model():

    model_name = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    )

    model.eval()

    return (
        tokenizer,
        model
    )


# ---------------------------------------------------------
# RAG FUNCTIONS
# ---------------------------------------------------------

def retrieve_rag_chunks(
    question,
    top_k=4
):

    embedding_model = (
        load_embedding_model()
    )

    rag_index = (
        load_faiss_index()
    )

    chunks = (
        load_rag_chunks()
    )

    query_embedding = (
        embedding_model.encode(
            [
                question
            ],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    scores, indices = (
        rag_index.search(
            query_embedding.astype(
                "float32"
            ),
            top_k
        )
    )

    results = (
        chunks.iloc[
            indices[0]
        ].copy()
    )

    results[
        "similarity_score"
    ] = scores[0]

    return results


def build_rag_messages(
    question,
    retrieved_chunks
):

    sections = []

    for i, (_, row) in enumerate(
        retrieved_chunks.iterrows(),
        start=1
    ):

        sections.append(
            f"""
SOURCE {i}
Source file: {row['source_file']}
Chunk ID: {row['chunk_id']}

{row['text']}
"""
        )

    context = "\n".join(
        sections
    )

    system_message = """
You are the RxReview discharge-support assistant.

Answer using only the retrieved source context.

Do not provide:
- patient-specific prescribing advice,
- medication dosing recommendations,
- recommendations to start, stop, or change medication.

If the source context does not explicitly support the requested information,
state that the retrieved RxReview sources do not provide enough information.

Keep answers concise and practical.
"""

    user_message = f"""
QUESTION:
{question}

SOURCE CONTEXT:
{context}

Answer using only the context above.
"""

    return [
        {
            "role":
                "system",
            "content":
                system_message
        },
        {
            "role":
                "user",
            "content":
                user_message
        }
    ]


def answer_rag_question(
    question,
    top_k=4,
    min_retrieval_score=0.40
):

    retrieved = (
        retrieve_rag_chunks(
            question,
            top_k
        )
    )

    top_score = (
        retrieved[
            "similarity_score"
        ].max()
        if len(
            retrieved
        ) > 0
        else 0
    )

    if (
        top_score
        <
        min_retrieval_score
    ):

        answer = (
            "The retrieved RxReview sources "
            "do not provide enough information "
            "to answer this question."
        )

        return (
            answer,
            retrieved
        )

    tokenizer, model = (
        load_generation_model()
    )

    messages = (
        build_rag_messages(
            question,
            retrieved
        )
    )

    model_inputs = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
    )

    model_inputs = {
        key:
            value.to(
                model.device
            )
        for key, value
        in model_inputs.items()
    }

    input_length = (
        model_inputs[
            "input_ids"
        ].shape[-1]
    )

    with torch.no_grad():

        generated_ids = (
            model.generate(
                **model_inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.05
            )
        )

    new_tokens = (
        generated_ids[
            0,
            input_length:
        ]
    )

    answer = (
        tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )
    )

    return (
        answer.strip(),
        retrieved
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title(
    "💊 RxReview"
)

st.subheader(
    "Pharmacist-Led Diabetes "
    "Discharge Review Decision Support"
)

st.caption(
    "Prioritize limited pharmacist capacity "
    "using readmission risk, medication complexity, "
    "cost/value analysis, and grounded discharge guidance."
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7
) = st.tabs([
    "Dashboard",
    "Review Queue",
    "Model Comparison",
    "Cost & Capacity",
    "Discharge Guidance",
    "Responsible Use",
    "About"
])


# ---------------------------------------------------------
# TAB 1 — DASHBOARD
# ---------------------------------------------------------

with tab1:

    st.header(
        "RxReview Dashboard"
    )

    st.caption(
        "Dashboard performance uses grouped out-of-fold "
        "predictions so the retrospective evaluation is "
        "not based on the final model's training predictions."
    )

    capacity = (
        capacity_selector(
            "Pharmacist Review Capacity",
            "dashboard_capacity_selector"
        )
    )

    selected = (
        select_by_capacity(
            priority_df,
            "rxreview_priority_score",
            capacity
            /
            100
        )
    )

    total_readmissions = int(
        priority_df[
            "readmit_30"
        ].sum()
    )

    captured = int(
        selected[
            "readmit_30"
        ].sum()
    )

    capture_rate = (
        captured
        /
        total_readmissions
        if total_readmissions > 0
        else 0
    )

    lift = (
        capture_rate
        /
        (
            capacity
            /
            100
        )
        if capacity > 0
        else 0
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    col1.metric(
        "Patients Reviewed",
        f"{len(selected):,}"
    )

    col2.metric(
        "Readmissions Captured",
        f"{captured:,}"
    )

    col3.metric(
        "Capture Rate",
        f"{capture_rate:.1%}"
    )

    col4.metric(
        "Lift vs Random",
        f"{lift:.2f}×"
    )

    st.markdown(
        "### Explore Capacity Tradeoffs"
    )

    visual_choice = (
        st.radio(
            "Choose a visualization",
            options=[
                "Capacity vs Capture",
                "Lift vs Random",
                "Patient Volume",
                "Net Value"
            ],
            horizontal=True,
            key="dashboard_visual"
        )
    )

    if (
        visual_choice
        ==
        "Capacity vs Capture"
    ):

        st.caption(
            "Shows the percentage of all observed "
            "30-day readmissions captured as pharmacist "
            "review capacity increases."
        )

        st.line_chart(
            capacity_summary_df,
            x="Capacity %",
            y="Capture Rate %"
        )

    elif (
        visual_choice
        ==
        "Lift vs Random"
    ):

        st.caption(
            "Lift shows how many times more readmissions "
            "RxReview captures than random selection at "
            "the same capacity."
        )

        st.line_chart(
            capacity_summary_df,
            x="Capacity %",
            y="Lift vs Random"
        )

    elif (
        visual_choice
        ==
        "Patient Volume"
    ):

        st.caption(
            "Shows the operational tradeoff between "
            "the number of patients reviewed and the "
            "number of observed readmissions captured."
        )

        st.line_chart(
            capacity_summary_df,
            x="Capacity %",
            y=[
                "Patients Reviewed",
                "Readmissions Captured"
            ]
        )

    else:

        st.caption(
            "Net value uses the baseline scenario: "
            "$75 pharmacist hourly cost, 30-minute review, "
            "$15,000 per readmission, 25% potentially "
            "preventable, and 20% intervention effectiveness."
        )

        st.line_chart(
            capacity_summary_df,
            x="Capacity %",
            y="Net Value"
        )


# ---------------------------------------------------------
# TAB 2 — REVIEW QUEUE
# ---------------------------------------------------------

with tab2:

    st.header(
        "Patient Review Queue"
    )

    st.success(
        "Live scoring active: this queue is regenerated "
        "from the saved XGBoost pipeline and isotonic "
        "calibrator loaded by the deployed app."
    )

    st.caption(
        "The final XGBoost model was fitted on all eligible "
        "project encounters for deployment/reuse. "
        "Formal performance estimates are shown separately "
        "using held-out or out-of-fold predictions."
    )

    queue_capacity = (
        capacity_selector(
            "Queue Capacity",
            "queue_capacity_selector"
        )
    )

    queue_df = (
        select_by_capacity(
            live_priority_df,
            "rxreview_priority_score",
            queue_capacity
            /
            100
        )
    )

    st.write(
        f"{queue_capacity}% capacity selects "
        f"{len(queue_df):,} patients for pharmacist review."
    )

    # -----------------------------------------------------
    # Define columns to display
    # -----------------------------------------------------

    display_cols = [
        col
        for col in [
            "patient_nbr",
            "xgb_calibrated_readmission_risk",
            "medication_complexity_score",
            "medication_complexity_weight",
            "rxreview_priority_score",
            "rxreview_priority_score_100",
            "medication_complexity_level",
            "rxreview_priority_level"
        ]
        if col in queue_df.columns
    ]

    # -----------------------------------------------------
    # Allow user to view either highest-priority patients
    # or patients closest to the selected capacity cutoff
    # -----------------------------------------------------

    queue_view = st.radio(
        "Queue View",
        options=[
            "Top 100 Highest Priority",
            "100 Patients Near Capacity Cutoff"
        ],
        horizontal=True,
        key="queue_view"
    )

    # Ensure queue is ordered from highest to lowest priority
    queue_df = (
        queue_df
        .sort_values(
            "rxreview_priority_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # Add operational queue rank
    queue_df[
        "Queue Rank"
    ] = range(
        1,
        len(queue_df) + 1
    )

    # -----------------------------------------------------
    # TOP 100 VIEW
    # -----------------------------------------------------

    if (
        queue_view
        ==
        "Top 100 Highest Priority"
    ):

        display_queue = (
            queue_df
            .head(
                100
            )
            .copy()
        )

        st.caption(
            f"Showing ranks 1–{len(display_queue):,} "
            f"of {len(queue_df):,} selected patients."
        )

    # -----------------------------------------------------
    # CAPACITY CUTOFF VIEW
    # -----------------------------------------------------

    else:

        start_idx = max(
            0,
            len(queue_df)
            -
            100
        )

        display_queue = (
            queue_df
            .iloc[
                start_idx:
            ]
            .copy()
        )

        first_rank = int(
            display_queue[
                "Queue Rank"
            ].iloc[
                0
            ]
        )

        last_rank = int(
            display_queue[
                "Queue Rank"
            ].iloc[
                -1
            ]
        )

        st.caption(
            f"Showing ranks {first_rank:,}–{last_rank:,}, "
            f"the patients closest to the "
            f"{queue_capacity}% capacity cutoff."
        )

    # -----------------------------------------------------
    # Display queue
    # -----------------------------------------------------

    queue_display_cols = [
        "Queue Rank"
    ] + [
        col
        for col in display_cols
        if col != "Queue Rank"
    ]

    st.dataframe(
        display_queue[
            queue_display_cols
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The queue is ranked by the live RxReview Priority Score. "
        "Scores support prioritization and do not replace "
        "clinical judgment."
    )

# ---------------------------------------------------------
# TAB 3 — MODEL COMPARISON
# ---------------------------------------------------------

with tab3:

    st.header(
        "Model & Priority-Method Comparison"
    )

    st.write(
        "RxReview compared traditional tabular models "
        "with the required fine-tuned classifier. "
        "Because pharmacist capacity is limited, "
        "ranking and capture at fixed capacity are "
        "more useful than accuracy alone."
    )

    st.markdown(
        "### Grouped Cross-Validated Tabular Models"
    )

    model_metrics = (
        pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],
            "ROC-AUC": [
                0.663897,
                0.673194,
                0.675042
            ],
            "PR-AUC": [
                0.214809,
                0.223132,
                0.229541
            ],
            "Precision": [
                0.181631,
                0.314075,
                0.187852
            ],
            "Recall": [
                0.545342,
                0.135496,
                0.565582
            ],
            "F1": [
                0.272502,
                0.189318,
                0.282031
            ]
        })
    )

    st.dataframe(
        model_metrics.style.format({
            "ROC-AUC":
                "{:.3f}",
            "PR-AUC":
                "{:.3f}",
            "Precision":
                "{:.3f}",
            "Recall":
                "{:.3f}",
            "F1":
                "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Among the grouped cross-validated tabular models, "
        "XGBoost provided the strongest overall ROC-AUC, "
        "PR-AUC, recall, and F1 combination."
    )

    st.markdown(
        "### Fine-Tuned DistilBERT — Same-Test Comparison"
    )

    same_test_metrics = (
        pd.DataFrame({
            "Model": [
                "XGBoost",
                "Fine-Tuned DistilBERT"
            ],
            "ROC-AUC": [
                0.677364,
                0.648281
            ],
            "PR-AUC": [
                0.223162,
                0.208289
            ],
            "10% Readmissions Captured": [
                402,
                369
            ],
            "10% Capture Rate": [
                0.242461,
                0.222557
            ],
            "10% Lift": [
                2.424608,
                2.225573
            ]
        })
    )

    st.dataframe(
        same_test_metrics.style.format({
            "ROC-AUC":
                "{:.3f}",
            "PR-AUC":
                "{:.3f}",
            "10% Capture Rate":
                "{:.1%}",
            "10% Lift":
                "{:.2f}×"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The same-test comparison avoids mixing the "
        "fine-tuned model's held-out test results with "
        "the grouped out-of-fold results above. At 10% "
        "capacity, XGBoost captured 402 of 1,658 observed "
        "30-day readmissions versus 369 for DistilBERT."
    )

    st.markdown(
        "### Priority-Method Comparison"
    )

    required_method_cols = {
        "readmit_30",
        "xgb_calibrated_readmission_risk",
        "medication_complexity_score",
        "rxreview_priority_score"
    }

    if required_method_cols.issubset(
        priority_df.columns
    ):

        method_rows = []

        for capacity_pct in [
            5,
            10,
            15,
            20
        ]:

            for (
                label,
                score_col
            ) in [

                (
                    "Risk Only",
                    "xgb_calibrated_readmission_risk"
                ),

                (
                    "Medication Complexity Only",
                    "medication_complexity_score"
                ),

                (
                    "Combined RxReview",
                    "rxreview_priority_score"
                )
            ]:

                method_selected = (
                    select_by_capacity(
                        priority_df,
                        score_col,
                        capacity_pct
                        /
                        100
                    )
                )

                captured_method = int(
                    method_selected[
                        "readmit_30"
                    ].sum()
                )

                method_rows.append({

                    "Capacity %":
                        capacity_pct,

                    "Method":
                        label,

                    "Readmissions Captured %":
                        (
                            captured_method
                            /
                            total_readmissions
                            *
                            100
                            if total_readmissions > 0
                            else 0
                        )
                })

        method_df = (
            pd.DataFrame(
                method_rows
            )
        )

        method_pivot = (
            method_df
            .pivot(
                index="Capacity %",
                columns="Method",
                values="Readmissions Captured %"
            )
            .reset_index()
        )

        st.line_chart(
            method_pivot,
            x="Capacity %",
            y=[
                "Risk Only",
                "Medication Complexity Only",
                "Combined RxReview"
            ]
        )

        selected_capacity_for_method = min(
            st.session_state[
                "review_capacity_pct"
            ],
            20
        )

        method_at_capacity = (
            method_df[
                method_df[
                    "Capacity %"
                ]
                ==
                selected_capacity_for_method
            ][
                [
                    "Method",
                    "Readmissions Captured %"
                ]
            ]
            .copy()
        )

        st.caption(
            f"Comparison at "
            f"{selected_capacity_for_method}% capacity:"
        )

        st.dataframe(
            method_at_capacity.style.format({
                "Readmissions Captured %":
                    "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Risk-only ranking captures slightly more "
            "readmissions, while the combined RxReview "
            "score intentionally incorporates medication "
            "complexity to better align the queue with a "
            "pharmacist-led medication-focused intervention."
        )

    else:

        st.warning(
            "Priority-method comparison columns are "
            "not available in the deployed evaluation file."
        )


# ---------------------------------------------------------
# TAB 4 — COST & CAPACITY
# ---------------------------------------------------------

with tab4:

    st.header(
        "Cost & Capacity Simulator"
    )

    col1, col2 = (
        st.columns(
            2
        )
    )

    with col1:

        sim_capacity = (
            st.slider(
                "Review Capacity (%)",
                5,
                30,
                10,
                5
            )
        )

        hourly_cost = (
            st.number_input(
                "Pharmacist Hourly Cost ($)",
                min_value=0.0,
                value=75.0,
                step=5.0
            )
        )

        review_minutes = (
            st.slider(
                "Minutes per Review",
                10,
                60,
                30,
                5
            )
        )

    with col2:

        readmission_cost = (
            st.number_input(
                "Cost per Readmission ($)",
                min_value=0.0,
                value=15000.0,
                step=1000.0
            )
        )

        preventable_pct = (
            st.slider(
                "Potentially Preventable (%)",
                0,
                100,
                25,
                5
            )
        )

        effectiveness_pct = (
            st.slider(
                "Intervention Effectiveness (%)",
                0,
                100,
                20,
                5
            )
        )

    sim = (
        rxreview_value_simulator(
            priority_df,
            sim_capacity
            /
            100,
            hourly_cost,
            review_minutes,
            readmission_cost,
            preventable_pct
            /
            100,
            effectiveness_pct
            /
            100
        )
    )

    st.markdown(
        "### Scenario Results"
    )

    c1, c2, c3, c4 = (
        st.columns(
            4
        )
    )

    c1.metric(
        "Patients Reviewed",
        f"{sim['patients_selected']:,}"
    )

    c2.metric(
        "Pharmacist Hours",
        f"{sim['pharmacist_hours']:,.0f}"
    )

    c3.metric(
        "Readmissions Captured",
        f"{sim['captured_readmissions']:,}"
    )

    c4.metric(
        "Lift",
        f"{sim['lift']:.2f}×"
    )

    c5, c6, c7 = (
        st.columns(
            3
        )
    )

    c5.metric(
        "Intervention Cost",
        f"${sim['intervention_cost']:,.0f}"
    )

    c6.metric(
        "Gross Savings",
        f"${sim['gross_savings']:,.0f}"
    )

    c7.metric(
        "Net Value",
        f"${sim['net_value']:,.0f}"
    )

    st.metric(
        "Break-Even Effectiveness",
        (
            f"{sim['break_even_effectiveness']:.1%}"
        )
    )

    st.caption(
        "Financial outputs are scenario estimates "
        "based on adjustable assumptions, not observed "
        "savings from the UCI dataset."
    )


# ---------------------------------------------------------
# TAB 5 — DISCHARGE GUIDANCE
# ---------------------------------------------------------

with tab5:

    st.header(
        "Discharge Guidance Assistant"
    )

    st.write(
        "Ask questions about medication reconciliation, "
        "discharge planning, patient education, "
        "follow-up, care transitions, or "
        "readmission-reduction practices."
    )

    question = (
        st.text_area(
            "Question",
            placeholder=(
                "Example: What should hospitals do "
                "to reconcile medications at discharge?"
            )
        )
    )

    if st.button(
        "Get Grounded Guidance"
    ):

        if question.strip():

            with st.spinner(
                "Retrieving evidence and generating response..."
            ):

                answer, sources = (
                    answer_rag_question(
                        question
                    )
                )

            st.markdown(
                "### Answer"
            )

            st.write(
                answer
            )

            st.markdown(
                "### Retrieved Sources"
            )

            st.dataframe(
                sources[
                    [
                        "source_file",
                        "chunk_id",
                        "similarity_score"
                    ]
                ],
                use_container_width=True
            )

        else:

            st.warning(
                "Enter a question first."
            )


# ---------------------------------------------------------
# TAB 6 — RESPONSIBLE USE
# ---------------------------------------------------------

with tab6:

    st.header(
        "Responsible Use"
    )

    st.markdown(
        """
### Intended Use

RxReview is an educational decision-support prototype for
prioritizing patients with diabetes for additional
pharmacist-led discharge review when pharmacist capacity
is limited.

It is designed to support resource allocation and workflow
prioritization — not autonomous clinical decision-making.

### RxReview Should Not Be Used To

- determine whether a medication regimen is clinically correct,
- diagnose medication errors,
- recommend specific medications or doses,
- recommend starting, stopping, or changing medication,
- determine whether a particular readmission was medication-related,
- replace pharmacist, physician, nursing, or care-management judgment.

### Data and Generalizability

The underlying UCI Diabetes dataset reflects hospital
encounters from 1999–2008. Clinical practice, medication
use, coding, patient populations, and care-transition
workflows have changed since that period.

The prototype therefore requires validation using current,
local hospital data before operational use.

### Medication Complexity

The Medication Complexity Weight is a project-defined
operational measure. It is not a validated medication-safety
or clinical-complexity scale.

### Subgroup Performance

Retrospective subgroup analysis showed relatively similar
performance across gender and across the two largest race
groups, while smaller race groups were more variable.
Performance also varied by age, with weaker discrimination
among patients age 80 and older.

These findings do not establish that the model is fair or
unfair. A real implementation would require ongoing
subgroup calibration, performance monitoring, drift
detection, and equity review.

### Calibration and Model Monitoring

The prototype uses isotonic calibration developed from
out-of-fold model predictions. This is appropriate for the
project workflow but should be revalidated prospectively
before clinical deployment.

### Human Oversight

RxReview prioritizes patients for human review. Final
decisions about medication reconciliation, counseling,
follow-up, or clinical intervention remain with qualified
healthcare professionals.
"""
    )


# ---------------------------------------------------------
# TAB 7 — ABOUT
# ---------------------------------------------------------

with tab7:

    st.header(
        "About RxReview"
    )

    st.markdown(
        """
**RxReview** is a decision-support prototype designed to
help hospitals prioritize patients with diabetes for
pharmacist-led discharge review when pharmacist capacity
is limited.

The application combines:

- a saved XGBoost readmission-classification pipeline,
- isotonic probability calibration,
- calibrated 30-day readmission risk,
- medication complexity,
- the RxReview Pharmacist-Review Priority Score,
- capacity-based patient prioritization,
- traditional and fine-tuned model comparison,
- scenario-based cost/value analysis,
- and source-grounded AHRQ/CMS discharge guidance.

### Live Model Use

The Review Queue is regenerated from the saved XGBoost
pipeline and calibrator when the deployed application loads.

The Dashboard and formal evaluation sections continue to
use grouped out-of-fold predictions so model performance is
not reported from the final model's training predictions.

### Business Decision

RxReview supports the question:

**Which patients should receive pharmacist-led medication
reconciliation and discharge counseling when pharmacist
capacity is limited?**

### Important Boundary

RxReview prioritizes patients for additional human review.
It does not prescribe treatment or determine whether a
medication regimen is clinically correct.
"""
    )
