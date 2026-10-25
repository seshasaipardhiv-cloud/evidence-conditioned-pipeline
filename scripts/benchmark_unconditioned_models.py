"""
benchmark_unconditioned_models.py

Evaluates unconditioned baseline models (default Logistic Regression, default XGBoost/GB, default Random Forest)
directly on the raw/standardized dataset without the pipeline's evidence-conditioning constraints,
and compares them directly against our Evidence-Conditioned Pipeline and Ensemble across all 5 cohorts.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss

from backend.app.final_integration.cohort_evaluator import CohortBenchmarkEvaluator

def run_benchmark():
    evaluator = CohortBenchmarkEvaluator()
    seeds = [42, 100, 2026]
    
    cohort_specs = {
        "Cohort_A_Authoritative_Hancock": evaluator._build_hancock_cohort(),
        "Cohort_B_Unseen_Cardiac_Tabular": evaluator._build_cardiac_cohort(),
        "Cohort_C_Unseen_Derm_Image":      evaluator._build_image_cohort(),
        "Cohort_D_Unseen_Pathology_Text":  evaluator._build_text_cohort(),
        "Cohort_E_Unseen_Trimodal_Oncology": evaluator._build_trimodal_cohort(),
    }
    
    benchmark_results = {}
    
    for c_key, c_raw in cohort_specs.items():
        print(f"Running benchmark on {c_key}...")
        adapted = evaluator.adapter.adapt_dataset(c_raw["records"])
        mods = adapted["discovered_modalities"]
        y = np.array(adapted["targets"], dtype=int)
        
        # Prepare feature matrix X for unconditioned baseline
        if len(mods) > 1:
            # Multimodal features
            img_f = evaluator._load_image_features(adapted, seed=42)
            txt_f = evaluator._compute_tfidf_features(adapted, seed=42)
            tab_f = adapted["tabular_features"]
            if tab_f is None or tab_f.shape[0] == 0:
                tab_f = np.zeros((len(y), 1), dtype=np.float32)
            X = np.hstack([tab_f, img_f, txt_f])
        elif "tabular" in mods:
            X = adapted["tabular_features"]
        elif "image" in mods:
            X = evaluator._load_image_features(adapted, seed=42)
        else:
            X = evaluator._compute_tfidf_features(adapted, seed=42)
            
        cohort_model_metrics = {
            "unconditioned_lr": {"roc": [], "pr": [], "f1": [], "acc": [], "brier": []},
            "unconditioned_xgb": {"roc": [], "pr": [], "f1": [], "acc": [], "brier": []},
            "unconditioned_rf": {"roc": [], "pr": [], "f1": [], "acc": [], "brier": []},
        }
        
        for seed in seeds:
            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None)
            
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            # 1. Unconditioned Default Logistic Regression
            lr = LogisticRegression(random_state=seed, max_iter=500)
            lr.fit(X_tr_sc, y_tr)
            p_lr = lr.predict_proba(X_te_sc)[:, 1]
            pred_lr = (p_lr >= 0.5).astype(int)
            cohort_model_metrics["unconditioned_lr"]["roc"].append(roc_auc_score(y_te, p_lr))
            cohort_model_metrics["unconditioned_lr"]["pr"].append(average_precision_score(y_te, p_lr))
            cohort_model_metrics["unconditioned_lr"]["f1"].append(f1_score(y_te, pred_lr, zero_division=0))
            cohort_model_metrics["unconditioned_lr"]["acc"].append(accuracy_score(y_te, pred_lr))
            cohort_model_metrics["unconditioned_lr"]["brier"].append(brier_score_loss(y_te, p_lr))
            
            # 2. Unconditioned Default XGBoost / GradientBoosting
            xgb = GradientBoostingClassifier(random_state=seed)
            xgb.fit(X_tr_sc, y_tr)
            p_xgb = xgb.predict_proba(X_te_sc)[:, 1]
            pred_xgb = (p_xgb >= 0.5).astype(int)
            cohort_model_metrics["unconditioned_xgb"]["roc"].append(roc_auc_score(y_te, p_xgb))
            cohort_model_metrics["unconditioned_xgb"]["pr"].append(average_precision_score(y_te, p_xgb))
            cohort_model_metrics["unconditioned_xgb"]["f1"].append(f1_score(y_te, pred_xgb, zero_division=0))
            cohort_model_metrics["unconditioned_xgb"]["acc"].append(accuracy_score(y_te, pred_xgb))
            cohort_model_metrics["unconditioned_xgb"]["brier"].append(brier_score_loss(y_te, p_xgb))
            
            # 3. Unconditioned Default Random Forest
            rf = RandomForestClassifier(random_state=seed)
            rf.fit(X_tr_sc, y_tr)
            p_rf = rf.predict_proba(X_te_sc)[:, 1]
            pred_rf = (p_rf >= 0.5).astype(int)
            cohort_model_metrics["unconditioned_rf"]["roc"].append(roc_auc_score(y_te, p_rf))
            cohort_model_metrics["unconditioned_rf"]["pr"].append(average_precision_score(y_te, p_rf))
            cohort_model_metrics["unconditioned_rf"]["f1"].append(f1_score(y_te, pred_rf, zero_division=0))
            cohort_model_metrics["unconditioned_rf"]["acc"].append(accuracy_score(y_te, pred_rf))
            cohort_model_metrics["unconditioned_rf"]["brier"].append(brier_score_loss(y_te, p_rf))
            
        summary = {}
        for m_name, vals in cohort_model_metrics.items():
            summary[m_name] = {
                "roc_auc_mean": round(float(np.mean(vals["roc"])), 4),
                "roc_auc_std": round(float(np.std(vals["roc"])), 4),
                "pr_auc_mean": round(float(np.mean(vals["pr"])), 4),
                "f1_mean": round(float(np.mean(vals["f1"])), 4),
                "accuracy_mean": round(float(np.mean(vals["acc"])), 4),
                "brier_mean": round(float(np.mean(vals["brier"])), 4),
            }
        benchmark_results[c_key] = summary

    # Load canonical pipeline and ensemble metrics from final_results.json
    final_res_path = Path("evidence/final/submission/New/results/final_results.json")
    with open(final_res_path, encoding="utf-8") as f:
        final_results = json.load(f)
        
    for r in final_results:
        ck = r["cohort_name"]
        if ck in benchmark_results:
            benchmark_results[ck]["evidence_pipeline"] = {
                "model_name": r["selected_model"],
                "roc_auc_mean": r["roc_auc_mean"],
                "roc_auc_std": r["roc_auc_std"],
                "pr_auc_mean": r["pr_auc_mean"],
                "f1_mean": r["f1_mean"],
                "accuracy_mean": r["accuracy_mean"],
                "brier_mean": r["brier_score_mean"],
            }
            benchmark_results[ck]["ensemble"] = {
                "members": r.get("ensemble_members", []),
                "roc_auc_mean": r.get("ensemble_roc_auc_mean"),
                "f1_mean": r.get("ensemble_f1_mean"),
            }
            
    out_path = Path("evidence/final/submission/New/results/unconditioned_benchmark_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"\nSaved unconditioned benchmark results to {out_path}")
    print("\n" + "="*80)
    print("COMPARISON: UNCONDITIONED BASELINES vs EVIDENCE PIPELINE vs ENSEMBLE")
    print("="*80)
    for ck, data in benchmark_results.items():
        print(f"\n[{ck}]")
        print(f"  Unconditioned LR:       ROC={data['unconditioned_lr']['roc_auc_mean']:.4f}, F1={data['unconditioned_lr']['f1_mean']:.4f}")
        print(f"  Unconditioned XGBoost:  ROC={data['unconditioned_xgb']['roc_auc_mean']:.4f}, F1={data['unconditioned_xgb']['f1_mean']:.4f}")
        print(f"  Unconditioned RF:       ROC={data['unconditioned_rf']['roc_auc_mean']:.4f}, F1={data['unconditioned_rf']['f1_mean']:.4f}")
        ep = data.get("evidence_pipeline", {})
        print(f"  Evidence Pipeline ({ep.get('model_name')}): ROC={ep.get('roc_auc_mean'):.4f}, F1={ep.get('f1_mean'):.4f}")
        ens = data.get("ensemble", {})
        print(f"  Ensemble:               ROC={ens.get('roc_auc_mean'):.4f}, F1={ens.get('f1_mean'):.4f}")
        
    return benchmark_results

if __name__ == "__main__":
    run_benchmark()
