import streamlit as st
import pandas as pd
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & MODERN CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Skein | AI Fraud Ring & Velocity Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6C5CE7, #a29bfe, #00cec9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .evidence-box {
        background: rgba(108, 92, 231, 0.08);
        border-left: 4px solid #6C5CE7;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin-top: 10px;
        margin-bottom: 15px;
        color: #f1f2f6;
    }
    .badge-block {
        background-color: #ff4757;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-review {
        background-color: #ffa502;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-approve {
        background-color: #2ed573;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA CACHING & LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    scores_df = pd.read_csv("fusion_scores.csv")
    amts_df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "TransactionAmt", "DeviceType", "DeviceInfo"])
    df = scores_df.merge(amts_df, on="TransactionID", how="left")
    
    with open("graph_clusters.json", "r") as f:
        cluster_data = json.load(f)
        
    cluster_meta = {int(k): v for k, v in cluster_data["cluster_meta"].items()}
    card1_to_cid = {int(k): v for k, v in cluster_data["card1_to_cid"].items()}
    
    # Reverse cluster mapping: cid -> list of card1s
    cid_to_cards = {}
    for card, cid in card1_to_cid.items():
        cid_to_cards.setdefault(cid, []).append(card)
        
    return df, cluster_meta, card1_to_cid, cid_to_cards

df, cluster_meta, card1_to_cid, cid_to_cards = load_all_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: OPERATING THRESHOLD & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Risk Policy & Thresholds")

threshold_mode = st.sidebar.radio(
    "Operating Cutoff Tier:",
    options=["Top 5% (⭐ Recommended)", "Top 1% (Strict Auto-Block)", "Top 10% (Aggressive Review)", "Custom Percentile"],
    index=0
)

if "Top 1%" in threshold_mode:
    pct_cutoff = 1.0
elif "Top 5%" in threshold_mode:
    pct_cutoff = 5.0
elif "Top 10%" in threshold_mode:
    pct_cutoff = 10.0
else:
    pct_cutoff = st.sidebar.slider("Custom Top % Riskiest:", min_value=0.5, max_value=15.0, value=5.0, step=0.5)

score_threshold = float(np.percentile(df["final_risk_score"], 100 - pct_cutoff))
top1_cutoff = float(np.percentile(df["final_risk_score"], 99))
st.sidebar.caption(f"🎯 Score Cutoff: **≥ {score_threshold:.4f}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Quick-Pick Demo Cases")
demo_picks = {
    "Select a demo case...": None,
    "🔥 Coordinated Ring Member (Txn 3577541, Ring #0)": 3577541,
    "⚡ Recovered Clique Member (Txn 3577548, Ring #1)": 3577548,
    "🌐 7-Card Louvain Ring (Txn 3577554, Ring #2)": 3577554,
    "📈 Pure Velocity Anomaly (Txn 3577646, Ring #20)": 3577646,
    "✅ Clean Legitimate Transaction (Txn 3561399)": 3561399,
}
demo_selection = st.sidebar.selectbox("Jump to known case:", options=list(demo_picks.keys()))
demo_txn_id = demo_picks[demo_selection]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About Skein:**
- **Track 02:** AI Risk Manager
- **Engine 1:** XGBoost Velocity Model
- **Engine 2:** Louvain + Clique Graph Ring Detector
- **Fusion:** $0.50 \\times \\text{Vel} + 0.50 \\times \\text{Graph}$
""")

# -----------------------------------------------------------------------------
# 4. SUMMARY METRICS & OPERATIONAL STATUS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ Skein: Fraud Ring & Velocity Risk Manager</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Live Evaluation Dashboard • Temporal Entity Graph + Past-Only Rolling Velocity Fusion</div>', unsafe_allow_html=True)

# Calculate operational stats for the current threshold
total_txns = len(df)
total_fraud = int(df["isFraud"].sum())
flagged_df = df[df["final_risk_score"] >= score_threshold]
n_flagged = len(flagged_df)
tp_count = int(flagged_df["isFraud"].sum())
fp_count = n_flagged - tp_count
precision = tp_count / n_flagged if n_flagged > 0 else 0.0
recall = tp_count / total_fraud if total_fraud > 0 else 0.0
loss_avoided = float(flagged_df[flagged_df["isFraud"] == 1]["TransactionAmt"].sum())
review_cost = fp_count * 50

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
with kpi1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_txns:,}</div><div class="metric-lbl">Total Transactions</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#a29bfe;">Top {pct_cutoff:.1f}%</div><div class="metric-lbl">Operating Tier</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ff7675;">{n_flagged:,}</div><div class="metric-lbl">Flagged ({100*n_flagged/total_txns:.1f}%)</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#55efc4;">{precision:.1%}</div><div class="metric-lbl">Precision</div></div>', unsafe_allow_html=True)
with kpi5:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#74b9ff;">{recall:.1%}</div><div class="metric-lbl">Fraud Recall</div></div>', unsafe_allow_html=True)
with kpi6:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ffeaa7;">₹{loss_avoided:,.0f}</div><div class="metric-lbl">Est. Loss Avoided</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN WORKSPACE: TRANSACTION EXPLORER & DEEP DIVE
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("### 📋 Transaction Explorer")
    
    # Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        view_filter = st.selectbox(
            "Filter Queue By:",
            options=["Flagged Transactions (Above Cutoff)", "Injected Ground-Truth Rings", "Graph Cluster Members Only", "All Transactions"]
        )
    with f_col2:
        search_txn = st.text_input("Search Transaction ID:", value=str(demo_txn_id) if demo_txn_id else "")

    if search_txn.strip().isdigit():
        target_id = int(search_txn.strip())
        display_df = df[df["TransactionID"] == target_id]
        if display_df.empty:
            st.warning(f"Transaction ID {target_id} not found. Showing filtered list.")
            display_df = df[df["final_risk_score"] >= score_threshold].head(50)
    else:
        if view_filter == "Flagged Transactions (Above Cutoff)":
            display_df = df[df["final_risk_score"] >= score_threshold].head(50)
        elif view_filter == "Injected Ground-Truth Rings":
            display_df = df[df["is_injected_ring"] == True].head(50)
        elif view_filter == "Graph Cluster Members Only":
            display_df = df[df["graph_cluster_id"] >= 0].head(50)
        else:
            display_df = df.head(50)

    # Assign risk badge labels
    def get_tier_label(score):
        if score >= top1_cutoff:
            return "🔴 Auto-Block"
        elif score >= score_threshold:
            return "🟡 Review"
        else:
            return "🟢 Approve"

    table_df = display_df[[
        "TransactionID", "card1", "TransactionAmt", "velocity_score", "graph_score", "final_risk_score", "isFraud"
    ]].copy()
    table_df["Action"] = table_df["final_risk_score"].apply(get_tier_label)
    table_df["TransactionAmt"] = table_df["TransactionAmt"].apply(lambda x: f"₹{x:,.2f}")
    table_df["velocity_score"] = table_df["velocity_score"].apply(lambda x: f"{x:.3f}")
    table_df["graph_score"] = table_df["graph_score"].apply(lambda x: f"{x:.3f}")
    table_df["final_risk_score"] = table_df["final_risk_score"].apply(lambda x: f"{x:.3f}")
    table_df.rename(columns={
        "TransactionID": "Txn ID", "card1": "Card Entity", "TransactionAmt": "Amount",
        "velocity_score": "Vel Score", "graph_score": "Graph Score", "final_risk_score": "Risk Score",
        "isFraud": "Ground Truth"
    }, inplace=True)

    st.dataframe(table_df, width="stretch", height=450)
    
    # Transaction selector
    selected_txn_id = st.selectbox(
        "Select Transaction ID for Investigation:",
        options=display_df["TransactionID"].tolist(),
        index=0 if not display_df.empty else None
    )

with col_right:
    st.markdown("### 🔬 Entity Investigation & Evidence")
    
    if selected_txn_id:
        row = df[df["TransactionID"] == selected_txn_id].iloc[0]
        
        # Risk Badge & Header Info
        risk = float(row["final_risk_score"])
        vel = float(row["velocity_score"])
        graph = float(row["graph_score"])
        card1 = int(row["card1"])
        amt = float(row["TransactionAmt"])
        is_fraud = bool(row["isFraud"])
        is_ring = bool(row["is_injected_ring"])
        ring_id = int(row["ring_id"]) if is_ring else None
        cid = int(row["graph_cluster_id"])
        
        c_badge, c_info = st.columns([0.4, 0.6])
        with c_badge:
            if risk >= top1_cutoff:
                st.markdown('<span class="badge-block">POLICY: AUTO-BLOCK</span>', unsafe_allow_html=True)
            elif risk >= score_threshold:
                st.markdown('<span class="badge-review">POLICY: MANUAL REVIEW</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-approve">POLICY: AUTO-APPROVE</span>', unsafe_allow_html=True)
        with c_info:
            if is_ring:
                st.markdown(f"**Validation:** Injected Fraud Ring #{ring_id}")
            elif is_fraud:
                st.markdown(f"**Validation:** Natural Dataset Fraud (1)")
            else:
                st.markdown(f"**Validation:** Clean Transaction (0)")
                
        # Metric score bars
        st.markdown(f"**Final Risk Score:** `{risk:.4f}`")
        st.progress(min(max(risk, 0.0), 1.0))
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Velocity Score", f"{vel:.3f}")
        with s2:
            st.metric("Graph Ring Score", f"{graph:.3f}")
        with s3:
            st.metric("Amount", f"₹{amt:,.2f}")
            
        # Evidence Trail
        st.markdown("**Automated Audit Trail / Evidence:**")
        st.markdown(f'<div class="evidence-box">📝 {row["evidence"]}</div>', unsafe_allow_html=True)
        
        # NetworkX Graph Visualization
        st.markdown("#### 🕸️ Coordinated Entity Graph")
        
        if cid >= 0 and cid in cluster_meta:
            meta = cluster_meta[cid]
            cluster_cards = cid_to_cards.get(cid, [card1])
            combos_str = ", ".join(meta["combos"])
            
            st.caption(f"Cluster **#{cid}** ({meta['source']}) • **{meta['size']} Accounts** • Density: **{meta['density']:.2f}**")
            st.caption(f"Shared Device Fingerprint: `{combos_str}`")
            
            # Draw NetworkX Graph
            G = nx.Graph()
            for c in cluster_cards:
                G.add_node(c)
            # Create edges between cluster cards
            for i in range(len(cluster_cards)):
                for j in range(i + 1, len(cluster_cards)):
                    G.add_edge(cluster_cards[i], cluster_cards[j])
                    
            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            
            pos = nx.spring_layout(G, seed=42)
            node_colors = ["#ff4757" if n == card1 else "#6c5ce7" for n in G.nodes()]
            node_sizes = [700 if n == card1 else 450 for n in G.nodes()]
            
            nx.draw_networkx_edges(G, pos, edge_color="#a29bfe", alpha=0.4, width=1.5, ax=ax)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="#ffffff", linewidths=1.2, ax=ax)
            
            labels = {n: f"card1\n{n}" if n == card1 else str(n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#ffffff", font_family="sans-serif", ax=ax)
            
            # Legend
            sel_patch = mpatches.Patch(color='#ff4757', label=f'Current Card ({card1})')
            peer_patch = mpatches.Patch(color='#6c5ce7', label='Ring Peer Card')
            ax.legend(handles=[sel_patch, peer_patch], loc="upper right", fontsize=8, facecolor="#1e222d", edgecolor="none", labelcolor="#ffffff")
            
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ℹ️ No multi-entity graph ring detected for this card. Transaction evaluated on rolling velocity and historical baseline.")
    else:
        st.info("Please select a transaction to view its entity graph and evidence trail.")
