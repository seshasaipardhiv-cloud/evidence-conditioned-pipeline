from backend.app.final_integration.cohort_evaluator import CohortBenchmarkEvaluator

evaluator = CohortBenchmarkEvaluator()
results = evaluator.evaluate_all_cohorts()
print("\n" + "="*80)
print("EVALUATION RESULTS ACROSS ALL 5 COHORTS:")
print("="*80)
for k, v in results.items():
    m = v['multi_seed_metrics']
    ens = v['ensemble_metrics']
    model_name = v['selected_components'].get('tabular_model', {}).get('selected_name', '')
    if not model_name:
        model_name = v['selected_components'].get('image_model', {}).get('selected_name', '')
    if not model_name:
        model_name = v['selected_components'].get('text_model', {}).get('selected_name', '')
    if not model_name:
        model_name = "Multimodal Pipeline"
    print(f"{k} ({model_name}):")
    print(f"  Candidate: ROC-AUC={m['roc_auc_mean']:.4f} +/- {m['roc_auc_std']:.4f}, PR-AUC={m['pr_auc_mean']:.4f}, F1={m['f1_mean']:.4f}, Acc={m['accuracy_mean']:.4f}")
    print(f"  Ensemble:  ROC-AUC={ens['roc_auc_mean']:.4f}, F1={ens['f1_mean']:.4f}")
print("="*80)
