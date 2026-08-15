from typing import List, Optional
from backend.app.stage2.models import (
    PaperRecord, EvidenceClaim, Mechanism,
    ExperimentRecord, AblationRecord,
    GraphNode, GraphRelationship, EvidenceGraph,
)


class GraphBuilder:
    def __init__(self):
        pass

    def build_graph(
        self,
        papers: List[PaperRecord],
        claims: List[EvidenceClaim],
        mechanisms: List[Mechanism],
        experiments: Optional[List[ExperimentRecord]] = None,
        ablations: Optional[List[AblationRecord]] = None,
    ) -> EvidenceGraph:
        """
        Builds the evidence graph with Stage 2A and Stage 2B nodes.

        Node types: Paper, Mechanism, EvidenceClaim, Modality, Experiment, Result, Ablation
        Relationship types:
          Paper -[reports]-> EvidenceClaim
          Paper -[has_experiment]-> Experiment
          EvidenceClaim -[uses]-> Mechanism
          EvidenceClaim -[uses_modality]-> Modality
          Experiment -[uses]-> Mechanism
          Experiment -[evaluated_on]-> Dataset (name only)
          Experiment -[measures]-> Metric
          Experiment -[produces]-> Result
          Experiment -[has_ablation]-> Ablation
          Result -[supported_by]-> EvidenceClaim
          Mechanism -[supported_by]-> EvidenceClaim
        """
        nodes: List[GraphNode] = []
        relationships: List[GraphRelationship] = []
        _node_ids: set = set()

        def add_node(node: GraphNode) -> None:
            if node.node_id not in _node_ids:
                nodes.append(node)
                _node_ids.add(node.node_id)

        # ── Paper nodes ───────────────────────────────────────────────────────
        for p in papers:
            add_node(GraphNode(
                node_id=p.paper_id,
                node_type="Paper",
                properties={
                    "title": p.title,
                    "doi": p.doi,
                    "pmid": p.pmid,
                    "year": p.publication_year,
                    "abstract_available": p.abstract_available,
                    "full_text_available": p.full_text_available,
                    "full_text_access_status": p.full_text_access_status.value,
                },
            ))

        # ── Mechanism nodes ───────────────────────────────────────────────────
        for m in mechanisms:
            add_node(GraphNode(
                node_id=m.mechanism_id,
                node_type="Mechanism",
                properties={
                    "canonical_name": m.canonical_name,
                    "category": m.category.value,
                    "mapping_status": m.mapping_status,
                    "role": m.role,
                },
            ))

        # ── EvidenceClaim nodes ───────────────────────────────────────────────
        for c in claims:
            add_node(GraphNode(
                node_id=c.evidence_id,
                node_type="EvidenceClaim",
                properties={
                    "claim_text": c.claim,
                    "evidence_status": c.evidence_status.value,
                    "source_scope": c.source_scope.value,
                    "contradiction_candidate": c.contradiction_candidate,
                    "paper_id": c.paper_id,
                },
            ))

            # Paper -> reports -> EvidenceClaim
            relationships.append(GraphRelationship(
                source_id=c.paper_id,
                target_id=c.evidence_id,
                relationship_type="reports",
            ))

            # EvidenceClaim -> uses -> Mechanism
            for mech_id in c.mechanisms:
                relationships.append(GraphRelationship(
                    source_id=c.evidence_id,
                    target_id=mech_id,
                    relationship_type="uses",
                ))

            # EvidenceClaim -> uses_modality -> Modality
            for mod in c.modalities:
                mod_id = f"mod_{mod}"
                add_node(GraphNode(
                    node_id=mod_id,
                    node_type="Modality",
                    properties={"name": mod},
                ))
                relationships.append(GraphRelationship(
                    source_id=c.evidence_id,
                    target_id=mod_id,
                    relationship_type="uses_modality",
                ))

        # ── Experiment nodes ──────────────────────────────────────────────────
        for exp in (experiments or []):
            add_node(GraphNode(
                node_id=exp.experiment_id,
                node_type="Experiment",
                properties={
                    "paper_id": exp.paper_id,
                    "task": exp.task,
                    "dataset": exp.dataset,
                    "sample_count": exp.sample_count,
                    "fusion_strategy": exp.fusion_strategy.value if exp.fusion_strategy else None,
                    "source_scope": exp.source_scope.value,
                },
            ))

            # Paper -> has_experiment -> Experiment
            relationships.append(GraphRelationship(
                source_id=exp.paper_id,
                target_id=exp.experiment_id,
                relationship_type="has_experiment",
            ))

            # Experiment -> evaluated_on -> Dataset node
            if exp.dataset:
                dataset_id = f"dataset_{exp.dataset.lower().replace(' ', '_')[:30]}"
                add_node(GraphNode(
                    node_id=dataset_id,
                    node_type="Dataset",
                    properties={"name": exp.dataset},
                ))
                relationships.append(GraphRelationship(
                    source_id=exp.experiment_id,
                    target_id=dataset_id,
                    relationship_type="evaluated_on",
                ))

            # Experiment -> measures -> Metric
            for metric in exp.evaluation_metrics:
                metric_id = f"metric_{metric.lower().replace(' ', '_')}"
                add_node(GraphNode(
                    node_id=metric_id,
                    node_type="Metric",
                    properties={"name": metric},
                ))
                relationships.append(GraphRelationship(
                    source_id=exp.experiment_id,
                    target_id=metric_id,
                    relationship_type="measures",
                ))

            # Experiment -> produces -> Result
            for idx, result in enumerate(exp.reported_results):
                result_id = f"result_{exp.experiment_id}_{idx}"
                add_node(GraphNode(
                    node_id=result_id,
                    node_type="Result",
                    properties={
                        "metric": result.metric,
                        "method_value": result.method_value,
                        "baseline_value": result.baseline_value,
                        "delta": result.delta,
                        "direction": result.direction,
                        "source_location": result.source_location,
                        "source_scope": result.source_scope.value,
                    },
                ))
                relationships.append(GraphRelationship(
                    source_id=exp.experiment_id,
                    target_id=result_id,
                    relationship_type="produces",
                ))

                # Result -> supported_by -> EvidenceClaim (link if same paper)
                for c in claims:
                    if c.paper_id == exp.paper_id:
                        relationships.append(GraphRelationship(
                            source_id=result_id,
                            target_id=c.evidence_id,
                            relationship_type="supported_by",
                        ))
                        break  # one link per result is sufficient

        # ── Ablation nodes ────────────────────────────────────────────────────
        for abl in (ablations or []):
            add_node(GraphNode(
                node_id=abl.ablation_id,
                node_type="Ablation",
                properties={
                    "paper_id": abl.paper_id,
                    "condition_removed": abl.condition_removed,
                    "source_location": abl.source_location,
                    "source_scope": abl.source_scope.value,
                    "direction": abl.result.direction if abl.result else None,
                },
            ))

            # Experiment -> has_ablation -> Ablation
            relationships.append(GraphRelationship(
                source_id=abl.parent_experiment_id,
                target_id=abl.ablation_id,
                relationship_type="has_ablation",
            ))

        return EvidenceGraph(nodes=nodes, relationships=relationships)
