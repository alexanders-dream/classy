# app.py
"""Main Streamlit application file for the Hierarchical Text Classifier."""

import streamlit as st
import pandas as pd
import os
import re # For sanitizing folder names
from pathlib import Path # Import Path
import traceback
from typing import List
# Import modules
import config
import utils
import ui_components
import hf_classifier
import llm_classifier

# --- Helper function to list saved models ---
def list_saved_hf_models(base_path: Path) -> List[str]:
    """Lists subdirectories in the base_path, assuming they are saved models."""
    base_path.mkdir(parents=True, exist_ok=True)
    try: return sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    except Exception as e: st.error(f"Error listing models in {base_path}: {e}"); return []

def sanitize_foldername(name: str) -> str:
    """Removes potentially problematic characters for folder names."""
    name = name.strip(); name = re.sub(r'[^\w\-\.]+', '_', name); name = name.strip('_.'); name = re.sub(r'_+', '_', name)
    return name if name else "unnamed_model"

# --- Page Configuration ---
st.set_page_config(page_title="AI Text Classifier", page_icon="🏷️", layout="wide", initial_sidebar_state="expanded")

# --- Load Environment Variables ---
from dotenv import load_dotenv; load_dotenv()

# --- Initialize Session State ---
if 'session_initialized' not in st.session_state:
    utils.init_session_state()
    if 'hf_model_save_name' not in st.session_state: st.session_state.hf_model_save_name = ""
    if 'hf_selected_load_model' not in st.session_state: st.session_state.hf_selected_load_model = None
    st.session_state.session_initialized = True

# --- Main App Title ---
st.title("🏷️ AI Hierarchical Text Classifier"); st.markdown("Classify text using HF models or LLMs.")

# --- Workflow Selection ---
st.sidebar.title("🛠️ Workflow"); workflow_options=["LLM Categorization","Hugging Face Model"]; selected_workflow=st.sidebar.radio("Choose Method:",options=workflow_options,key='selected_workflow',horizontal=True); st.sidebar.markdown("---")

# --- Display LLM Sidebar ---
if selected_workflow == "LLM Categorization": ui_components.display_llm_sidebar(); llm_ready = st.session_state.get('llm_client') is not None
else: llm_ready = False

# --- Main Application Tabs ---
tab_setup, tab_hierarchy, tab_classify, tab_results = st.tabs(["1. Data Setup","2. Hierarchy","3. Classification","4. Results"])

# === Tab 1: Data Setup ===
# (No changes needed in Tab 1)
with tab_setup:
    st.header("1. Data Upload and Column Selection"); st.markdown("Upload data & select text column.")
    col1, col2 = st.columns(2)
    with col1: # Prediction Data
        st.subheader("Data to Classify"); help_pred = "Upload CSV/Excel with text to categorize."
        uncategorized_file = st.file_uploader("Upload Unlabeled CSV/Excel", type=['csv','xlsx','xls'], key="uncat_uploader_main", help=help_pred)
        if uncategorized_file and uncategorized_file.file_id != st.session_state.get('uncategorized_file_key'):
            with st.spinner(f"Loading '{uncategorized_file.name}'..."): st.session_state.uncategorized_df = utils.load_data(uncategorized_file)
            st.session_state.uncategorized_file_key = uncategorized_file.file_id; st.session_state.uncat_text_col = None; st.session_state.results_df = None
            st.session_state.app_stage = 'file_uploaded' # No rerun needed here
        if st.session_state.uncategorized_df is not None:
            st.success(f"Prediction data loaded ({st.session_state.uncategorized_df.shape[0]} rows).")
            with st.expander("Preview"): st.dataframe(st.session_state.uncategorized_df.head())
            st.markdown("**Select Text Column:**"); df_u = st.session_state.uncategorized_df; df_u.columns = df_u.columns.astype(str); cols_u=[""]+df_u.columns.tolist()
            curr_u_col=st.session_state.uncat_text_col; def_u_idx=0;
            try:
                if curr_u_col in cols_u: def_u_idx=cols_u.index(curr_u_col)
                elif curr_u_col: def_u_idx=0; st.session_state.uncat_text_col=None
            except ValueError: pass
            sel_u_col = st.selectbox("Select:",options=cols_u,index=def_u_idx,key="uncat_text_select_main",label_visibility="collapsed")
            if sel_u_col and sel_u_col != st.session_state.uncat_text_col: st.session_state.uncat_text_col=sel_u_col; st.session_state.app_stage='column_selected' # No rerun needed
            elif sel_u_col: st.caption(f"Using: **'{sel_u_col}'**")
        else: st.info("Upload data to classify.")
    with col2: # Training Data (HF Only)
        if selected_workflow=="Hugging Face Model":
            st.subheader("Training Data (HF)"); help_train="Labeled data (text & hierarchy cols)."
            categorized_file = st.file_uploader("Upload Labeled CSV/Excel", type=['csv','xlsx','xls'], key="cat_uploader_main", help=help_train)
            if categorized_file and categorized_file.file_id != st.session_state.get('categorized_file_key'):
                with st.spinner(f"Loading '{categorized_file.name}'..."): st.session_state.categorized_df = utils.load_data(categorized_file)
                st.session_state.categorized_file_key = categorized_file.file_id; st.session_state.cat_text_col=None
                for lvl in config.HIERARCHY_LEVELS: st.session_state[f'cat_{lvl.lower()}_col']=None
                st.session_state.hf_model_ready=False; st.session_state.hf_model=None; st.session_state.hf_tokenizer=None; st.session_state.hf_label_map=None; st.session_state.hf_rules=pd.DataFrame(columns=['Label','Keywords','Confidence Threshold','Custom Keywords']) # No rerun needed
            if st.session_state.categorized_df is not None:
                st.success(f"Training data loaded ({st.session_state.categorized_df.shape[0]} rows).")
                with st.expander("Preview"): st.dataframe(st.session_state.categorized_df.head())
                with st.form("hf_col_sel_form"):
                    st.markdown("**Select HF Training Cols:**"); df_c=st.session_state.categorized_df; df_c.columns=df_c.columns.astype(str); avail_c=df_c.columns.tolist();
                    curr_c_txt=st.session_state.cat_text_col; def_c_idx=0;
                    try:
                        if curr_c_txt in avail_c: def_c_idx=avail_c.index(curr_c_txt)
                        elif curr_c_txt: def_c_idx=0; st.session_state.cat_text_col=None
                    except ValueError: pass
                    sel_c_txt=st.selectbox("Text Col:", avail_c, index=def_c_idx, key="cat_text_sel_main")
                    st.markdown("Hierarchy Cols:"); sel_h_hf={}
                    for lvl in config.HIERARCHY_LEVELS:
                        k=f'cat_{lvl.lower()}_col'; cv=st.session_state.get(k,"(None)"); opts=["(None)"]+[c for c in avail_c if c!=sel_c_txt]
                        try: idx=opts.index(cv) if cv in opts else 0
                        except ValueError: idx=0
                        sel_h_hf[lvl]=st.selectbox(f"{lvl}:",opts,index=idx,key=f"{k}_sel_main")
                    sub_hf_cols=st.form_submit_button("Confirm")
                    if sub_hf_cols:
                        active_sels={l:c for l,c in sel_h_hf.items() if c and c!="(None)"}
                        if not sel_c_txt: st.warning("Select Text Col.")
                        elif not active_sels: st.warning("Select >=1 Hierarchy Col.")
                        elif len(list(active_sels.values()))!=len(set(list(active_sels.values()))): st.warning("Duplicate cols.")
                        else: st.session_state.cat_text_col=sel_c_txt; [st.session_state.update({f'cat_{l.lower()}_col':c}) for l,c in sel_h_hf.items()]; st.success("HF cols confirmed.")
            else: st.info("Upload labeled data for HF.")
        else: st.info("Training data only needed for HF.")


# === Tab 2: Hierarchy Definition ===
with tab_hierarchy:
    st.header("2. Define Classification Hierarchy")
    # Define default value before conditional blocks
    suggestion_possible = False

    if selected_workflow == "LLM Categorization":
        st.markdown("Define structure for LLM. Use editor, AI suggestion, or refine.")
        st.subheader("🤖 AI Hierarchy Suggestion")
        # Calculate suggestion_possible based on conditions
        suggestion_possible = (st.session_state.get('uncategorized_df') is not None and
                              st.session_state.get('uncat_text_col'))

        if suggestion_possible:
            data_col_name = st.session_state.uncat_text_col
            slider_enabled = False; max_slider_val = config.MIN_LLM_SAMPLE_SIZE; default_slider_val = config.MIN_LLM_SAMPLE_SIZE
            try:
                 if data_col_name in st.session_state.uncategorized_df.columns:
                    unique_texts = pd.to_numeric(st.session_state.uncategorized_df[data_col_name], errors='coerce').dropna().astype(str).unique()
                    unique_count = len(unique_texts); min_needed = config.MIN_LLM_SAMPLE_SIZE
                    max_slider_val = min(config.MAX_LLM_SAMPLE_SIZE, unique_count) if unique_count >= min_needed else min_needed
                    default_slider_val = min(config.DEFAULT_LLM_SAMPLE_SIZE, max_slider_val)
                    slider_enabled = llm_ready and (unique_count >= min_needed)
                    if not slider_enabled and llm_ready: st.warning(f"Need >= {min_needed} unique texts.")
                 else: st.warning(f"Col '{data_col_name}' not found.")
            except Exception as e: st.error(f"Suggest prep Error: {e}")

            sample_size = st.slider(f"Samples from '{data_col_name}':", config.MIN_LLM_SAMPLE_SIZE, max_slider_val, default_slider_val, 50, key="ai_sample_slider_main", help="More samples -> more context -> longer time.", disabled=not slider_enabled)
            if st.button("🚀 Generate Suggestion", key="generate_ai_hierarchy_main", type="primary", disabled=not slider_enabled):
                if llm_ready:
                    st.session_state.ai_suggestion_pending = None
                    sample_series = st.session_state.uncategorized_df[data_col_name].dropna().astype(str)
                    unique_list = sample_series.unique().tolist()
                    if len(unique_list) >= config.MIN_LLM_SAMPLE_SIZE:
                        actual_size = min(len(unique_list), sample_size)
                        sampled = pd.Series(unique_list).sample(actual_size, random_state=42).tolist() if len(unique_list) > actual_size else unique_list
                        suggestion = llm_classifier.generate_hierarchy_suggestion(st.session_state.llm_client, sampled)
                        if suggestion: st.session_state.ai_suggestion_pending = suggestion; st.success("✅ Suggestion generated!")
                        else: st.error("❌ Suggestion failed.")
                        st.rerun() # Rerun needed to show suggestion buttons
                    else: st.warning(f"Need >= {config.MIN_LLM_SAMPLE_SIZE} unique texts.")
                else: st.error("LLM Client not ready.")
        else: # suggestion_possible is False
             st.info("Upload prediction data & select column in Tab 1.")
        st.divider()
        st.subheader("✏️ Hierarchy Editor"); hierarchy_valid = ui_components.display_hierarchy_editor(key_prefix="llm")

    elif selected_workflow=="Hugging Face Model":
        st.info("HF hierarchy defined by columns in Tab 1.")
        active_hf_cols={l:st.session_state.get(f'cat_{l.lower()}_col') for l in config.HIERARCHY_LEVELS if st.session_state.get(f'cat_{l.lower()}_col') and st.session_state.get(f'cat_{l.lower()}_col')!="(None)"}
        if active_hf_cols: st.write("**HF Columns:**"); [st.write(f"- **{l}:** '{c}'") for l,c in active_hf_cols.items()]
        else: st.warning("Select HF columns in Tab 1.")

# === Tab 3: Run Classification ===
with tab_classify:
    st.header("3. Run Classification")

    if st.session_state.get('classification_just_completed', False):
        st.success("✅ Classification Complete! View results in the '4. View Results' tab.")

    # --- Hugging Face Workflow ---
    if selected_workflow == "Hugging Face Model":
        st.subheader("Train or Load Hugging Face Model")
        hf_cols_ready=(st.session_state.get('cat_text_col') and any(st.session_state.get(f'cat_{l.lower()}_col') and st.session_state.get(f'cat_{l.lower()}_col')!="(None)" for l in config.HIERARCHY_LEVELS))
        hf_data_ready=(st.session_state.get('categorized_df') is not None); hf_train_ready=hf_data_ready and hf_cols_ready

        # Option A: Train
        with st.container(border=True):
            st.markdown("**Option A: Train/Retrain HF Model**")
            if not hf_train_ready: st.warning("Setup Training Data & confirm columns in Tab 1.")
            c1,c2,c3=st.columns(3); train_dis=not hf_train_ready
            with c1: hf_mc=st.selectbox("Base Model:",["distilbert-base-uncased","bert-base-uncased","roberta-base"],0,disabled=train_dis,key="hf_m_sel")
            with c2: hf_ne=st.slider("Epochs:",1,10,3,1,disabled=train_dis,key="hf_e_sli")
            with c3: v_spl=st.slider("Val Split (%):",5,50,int(config.DEFAULT_VALIDATION_SPLIT*100),5,help="Validation %.",disabled=train_dis,key="hf_v_sli"); v_spl_r=v_spl/100.0

            train_btn_text = "🚀 Start HF Training" if not st.session_state.hf_model_ready else "🔄 Retrain HF Model"
            if st.button(train_btn_text, type="primary", disabled=train_dis):
                st.session_state.classification_just_completed = False
                # Reset state ONLY if retraining
                if st.session_state.hf_model_ready:
                    st.info("Clearing previous model state...")
                    st.session_state.update({'hf_model':None, 'hf_tokenizer':None, 'hf_label_map':None, 'hf_rules':pd.DataFrame(), 'hf_model_ready':False, 'hf_model_save_name':''})
                    # *** REMOVED the immediate st.rerun() here ***

                # Proceed with preparation and training logic
                h_map={l:st.session_state.get(f'cat_{l.lower()}_col') for l in config.HIERARCHY_LEVELS}; active_h={l:c for l,c in h_map.items() if c and c!="(None)"}
                if not st.session_state.cat_text_col or not active_h: st.error("Confirm HF columns.")
                else:
                    p_texts, p_labels = hf_classifier.prepare_hierarchical_training_data(st.session_state.categorized_df, st.session_state.cat_text_col, h_map)
                    if p_texts and p_labels:
                        st.info("Starting training process...") # Give user feedback before potentially long process
                        m, t, lm, r = hf_classifier.train_hf_model(p_texts, p_labels, hf_mc, hf_ne, v_spl_r)
                        if m and t and lm is not None:
                             st.session_state.update({'hf_model':m,'hf_tokenizer':t,'hf_label_map':lm,'hf_rules':r,'hf_model_ready':True,'hf_model_save_name':f"trained_{hf_mc.split('/')[-1]}"})
                             st.success("✅ HF Model trained!")
                             st.rerun() # Rerun AFTER training to update UI (like save section)
                        else: st.error("❌ Training failed.")
                    else: st.error("❌ Data prep failed.")

            # Save Section
            if st.session_state.hf_model_ready and st.session_state.hf_model:
                st.success(f"HF Model Ready: {st.session_state.get('hf_model_save_name', '(Untitled)')}")
                st.markdown("**Save Current Model:**"); save_n=st.text_input("Save Name:",value=st.session_state.get("hf_model_save_name",""),key="hf_save_n_in")
                if st.button("💾 Save Model", key="save_hf_b_ex"):
                    st.session_state.classification_just_completed=False; san_n=sanitize_foldername(save_n)
                    if not san_n or san_n=="unnamed_model": st.warning("Enter valid name.")
                    else:
                        save_d=config.SAVED_HF_MODELS_BASE_PATH/san_n
                        if save_d.exists(): st.warning(f"'{san_n}' exists. Overwriting.")
                        if hf_classifier.save_hf_model_artifacts(st.session_state.hf_model,st.session_state.hf_tokenizer,st.session_state.hf_label_map,st.session_state.hf_rules,str(save_d)): st.session_state.hf_model_save_name=san_n; st.success(f"Model saved: '{san_n}'")
                        else: st.error("Save failed.")
            elif not hf_train_ready and not st.session_state.hf_model_ready: pass

        # Option B: Load
        st.markdown("---");
        with st.container(border=True):
            st.markdown("**Option B: Load HF Model**"); st.caption(f"Path: `{config.SAVED_HF_MODELS_BASE_PATH}`"); saved_m=list_saved_hf_models(config.SAVED_HF_MODELS_BASE_PATH)
            if not saved_m: st.info("No saved models.")
            else:
                load_opts=[""]+saved_m; curr_idx=0; curr_name=st.session_state.get('hf_model_save_name')
                if curr_name in load_opts: curr_idx=load_opts.index(curr_name)
                load_sel=st.selectbox("Select Model:",options=load_opts,index=curr_idx,key="hf_load_sel",help="Choose folder.")
                load_dis=st.session_state.hf_model_ready or not load_sel
                if st.button("🔄 Load Model",disabled=load_dis):
                    st.session_state.classification_just_completed=False
                    if st.session_state.hf_model_ready: st.warning("Model active.")
                    elif load_sel:
                        load_d=config.SAVED_HF_MODELS_BASE_PATH/load_sel
                        with st.spinner(f"Loading '{load_sel}'..."): m,t,lm,r=hf_classifier.load_hf_model_artifacts(str(load_d))
                        if m and t and lm is not None: st.session_state.update({'hf_model':m,'hf_tokenizer':t,'hf_label_map':lm,'hf_rules':r,'hf_model_ready':True,'hf_model_save_name':load_sel}); st.success(f"✅ Loaded '{load_sel}'!"); st.rerun()
                        else: st.error("❌ Load failed.")
                    else: st.warning("Select model folder.")
            if st.session_state.hf_model_ready: st.success(f"Active Model: '{st.session_state.get('hf_model_save_name','Loaded')}'")

        # HF Rule Editor
        st.markdown("---"); st.subheader("Review & Edit HF Rules")
        if st.session_state.hf_model_ready and st.session_state.hf_rules is not None and not st.session_state.hf_rules.empty:
             if 'Label' in st.session_state.hf_rules.columns and 'Confidence Threshold' in st.session_state.hf_rules.columns:
                 with st.form("hf_rules_form"):
                    st.info("Edit Labels, Keywords (comma-sep), & Thresholds. Keywords force label."); rules_edit=st.session_state.hf_rules.copy()
                    if 'Keywords' not in rules_edit.columns: rules_edit['Keywords']='N/A'
                    rules_edit['Keywords']=rules_edit['Keywords'].fillna('N/A'); rules_edit['Confidence Threshold']=pd.to_numeric(rules_edit['Confidence Threshold'],errors='coerce').fillna(config.DEFAULT_HF_THRESHOLD).clip(0.05,0.95); rules_edit['Label']=rules_edit['Label'].astype(str)
                    edited_df=st.data_editor(rules_edit.sort_values('Label'), column_config={"Label":st.column_config.TextColumn("Label (Editable)"),"Keywords":st.column_config.TextColumn("Keywords (Editable)"),"Confidence Threshold":st.column_config.NumberColumn(min_value=0.05,max_value=0.95,step=0.01,format="%.3f")}, use_container_width=True,hide_index=True,num_rows="dynamic",key="hf_rules_edit_in_form")
                    submitted=st.form_submit_button("Save Rule Changes")
                    if submitted:
                        st.session_state.classification_just_completed=False
                        if isinstance(edited_df,pd.DataFrame):
                            if 'Label' not in edited_df.columns or edited_df['Label'].isnull().any() or (edited_df['Label']=='').any(): st.warning("Labels required.")
                            elif 'Confidence Threshold' not in edited_df.columns: st.warning("Threshold col missing.")
                            else:
                                edited_df['Confidence Threshold']=pd.to_numeric(edited_df['Confidence Threshold'],errors='coerce').fillna(config.DEFAULT_HF_THRESHOLD).clip(0.05,0.95); edited_df['Keywords']=edited_df['Keywords'].fillna('')
                                cols_comp=['Label','Keywords','Confidence Threshold']; curr_comp=st.session_state.hf_rules[cols_comp].sort_values('Label').reset_index(drop=True).astype(str); edit_comp=edited_df[cols_comp].sort_values('Label').reset_index(drop=True).astype(str)
                                if not curr_comp.equals(edit_comp): st.session_state.hf_rules=edited_df.copy(); st.success("✅ Rules saved!")
                                else: st.info("No changes.")
                        else: st.error("Editor error.")
             else: st.warning("Cannot edit rules: Core cols missing.")
        elif st.session_state.hf_model_ready: st.info("Model ready, no rules.")
        else: st.info("Train/load HF model first.")

        # Run HF Classification
        st.divider(); st.header("🚀 Run HF Classification")
        hf_ready=(st.session_state.hf_model_ready and st.session_state.uncategorized_df is not None and st.session_state.uncat_text_col is not None)
        if st.button("Classify using HF",type="primary",disabled=not hf_ready):
            st.session_state.classification_just_completed=False
            if st.session_state.uncat_text_col and st.session_state.uncat_text_col in st.session_state.uncategorized_df.columns:
                texts=st.session_state.uncategorized_df[st.session_state.uncat_text_col].fillna("").astype(str).tolist()
                if texts:
                    raw_lbls=hf_classifier.classify_texts_with_hf(texts,st.session_state.hf_model,st.session_state.hf_tokenizer,st.session_state.hf_label_map,st.session_state.hf_rules)
                    st.session_state.raw_predicted_labels=raw_lbls; parsed=utils.parse_predicted_labels_to_columns(raw_lbls); res_df=st.session_state.uncategorized_df.copy(); p_df=pd.DataFrame(parsed,index=res_df.index)
                    for c in config.HIERARCHY_LEVELS: res_df[f"HF_{c}"]=p_df.get(c)
                    res_df["HF_Raw_Labels"]=[', '.join(map(str,ls)) if ls else None for ls in raw_lbls]; st.session_state.results_df=res_df; st.session_state.app_stage='categorized'
                    st.session_state.classification_just_completed=True; st.rerun()
                else: st.warning("No text.")
            elif st.session_state.uncategorized_df: st.error(f"Pred col '{st.session_state.uncat_text_col}' missing.")
            else: st.error("Pred data missing.")
        elif not hf_ready: st.warning("Ensure HF model & data ready.")

    # --- LLM Workflow: Classification ---
    # (LLM Classification section remains the same)
    elif selected_workflow=="LLM Categorization":
        st.subheader("🚀 Run LLM Classification"); llm_ready_cls=(llm_ready and st.session_state.get('hierarchy_defined',False) and st.session_state.uncategorized_df is not None and st.session_state.uncat_text_col is not None)
        if not st.session_state.get('hierarchy_defined',False): st.warning("Define valid hierarchy in Tab 2.")
        if not llm_ready: st.warning("LLM Client not ready (sidebar).")
        if st.session_state.uncategorized_df is None or st.session_state.uncat_text_col is None: st.warning("Upload data & select column in Tab 1.")
        if st.button("Classify using LLM",type="primary",disabled=not llm_ready_cls):
             st.session_state.classification_just_completed=False; final_h=utils.build_hierarchy_from_df(st.session_state.hierarchy_df)
             if final_h and final_h.get('themes'):
                if st.session_state.uncat_text_col and st.session_state.uncat_text_col in st.session_state.uncategorized_df.columns:
                     res_llm=llm_classifier.classify_texts_with_llm(st.session_state.uncategorized_df,st.session_state.uncat_text_col,final_h,st.session_state.llm_client)
                     if res_llm is not None: st.session_state.results_df=res_llm; st.session_state.app_stage='categorized'; st.session_state.classification_just_completed=True; st.rerun()
                     else: st.error("LLM Classification failed.")
                else: st.error(f"Pred col '{st.session_state.uncat_text_col}' not found.")
             else: st.error("Cannot run LLM: Hierarchy invalid.")
        elif not llm_ready_cls: st.info("Complete setup for LLM.")


# === Tab 4: Results ===
# (Tab 4 code remains the same)
with tab_results:
    st.header("4. View Results")
    if st.session_state.get('classification_just_completed', False): st.info("✨ Displaying latest results."); st.session_state.classification_just_completed = False
    if st.session_state.get('results_df') is not None:
        res_df_copy=st.session_state.results_df.copy(); st.dataframe(res_df_copy,use_container_width=True); c1,c2=st.columns(2)
        with c1: csv=res_df_copy.to_csv(index=False).encode('utf-8'); st.download_button("📥 Download CSV",csv,'results.csv','text/csv',key='dl-csv')
        with c2: excel=utils.df_to_excel_bytes(res_df_copy); st.download_button("📥 Download Excel",excel,'results.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='dl-excel')
        st.divider()
        if selected_workflow=="Hugging Face Model" and st.session_state.get('raw_predicted_labels') is not None: st.subheader("📊 HF Stats"); 
        try: utils.display_hierarchical_stats(res_df_copy, prefix="HF_"); 
        except Exception as e: st.error(f"HF Stats Error: {e}") 
        
    elif selected_workflow=="LLM Categorization":
            st.subheader("📊 LLM Summary");
            try:
                total=len(res_df_copy); t_col, c_col, r_col = 'LLM_Theme','LLM_Category','LLM_Reasoning'; categorized=0
                if t_col in res_df_copy.columns and c_col in res_df_copy.columns: categorized=res_df_copy[(res_df_copy[t_col].notna()|res_df_copy[c_col].notna())&(res_df_copy[t_col]!='Error')].shape[0]
                errors=0
                if t_col in res_df_copy.columns: err_idx=set(res_df_copy[res_df_copy[t_col]=='Error'].index);
                if r_col in res_df_copy.columns: err_idx.update(set(res_df_copy[(res_df_copy[t_col].isna())&(res_df_copy[r_col].astype(str).str.contains("Error:",na=False))].index)); errors=len(err_idx)
                elif r_col in res_df_copy.columns: errors=res_df_copy[res_df_copy[r_col].astype(str).str.contains("Error:",na=False)].shape[0]
                st.metric("Total Rows",f"{total:,}"); st.metric("Categorized",f"{categorized:,}");
                if errors>0: st.metric("Errors",f"{errors:,}",delta=f"{errors}",delta_color="inverse")
                if t_col in res_df_copy.columns: st.markdown("**Themes:**"); counts=res_df_copy[t_col].value_counts().reset_index(); counts.columns=['Theme','Count']; st.dataframe(counts,use_container_width=True)
            except Exception as e: st.error(f"LLM Summary Error: {e}")
    else: st.info("Run classification in Tab 3.")

# --- Footer ---
st.sidebar.divider(); st.sidebar.caption(f"AI Classifier App v1.6") # Increment version

# --- Entry Point ---
if __name__=="__main__": pass