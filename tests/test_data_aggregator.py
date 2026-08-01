import pytest
from app.models.clinical_trials import NormalizedStudy, StudyIntervention
from app.models.schemas import VisualizationType
from app.services.query_analyzer import QueryIntentAnalysis
from app.services.data_aggregator import DataAggregator

@pytest.fixture
def sample_studies():
    return [
        NormalizedStudy(
            nct_id="NCT0001",
            brief_title="Study 1 of Pembrolizumab in Lung Cancer",
            start_year=2020,
            completion_year=2022,
            lead_sponsor="Merck Sharp & Dohme",
            lead_sponsor_class="INDUSTRY",
            conditions=["Lung Cancer"],
            interventions=[StudyIntervention(name="Pembrolizumab", type="DRUG")],
            phases=["Phase 3"],
            countries=["United States"],
            enrollment=300
        ),
        NormalizedStudy(
            nct_id="NCT0002",
            brief_title="Study 2 of Pembrolizumab in Melanoma",
            start_year=2020,
            completion_year=2023,
            lead_sponsor="Merck Sharp & Dohme",
            lead_sponsor_class="INDUSTRY",
            conditions=["Melanoma"],
            interventions=[StudyIntervention(name="Pembrolizumab", type="DRUG")],
            phases=["Phase 2"],
            countries=["United States", "Canada"],
            enrollment=150
        ),
        NormalizedStudy(
            nct_id="NCT0003",
            brief_title="NCI Trial for Lung Cancer",
            start_year=2021,
            completion_year=2024,
            lead_sponsor="National Cancer Institute",
            lead_sponsor_class="NIH",
            conditions=["Lung Cancer"],
            interventions=[StudyIntervention(name="Carboplatin", type="DRUG")],
            phases=["Phase 1"],
            countries=["United States"],
            enrollment=50
        )
    ]

def test_time_series_aggregation(sample_studies):
    aggregator = DataAggregator()
    analysis = QueryIntentAnalysis(
        intent="time_trend",
        recommended_visualization=VisualizationType.TIME_SERIES,
        suggested_title="Time Series Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(sample_studies, analysis, {"term": "Pembrolizumab"})

    assert spec.type == VisualizationType.TIME_SERIES
    assert len(spec.data) == 2  # Years 2020 and 2021
    assert spec.data[0]["year"] == 2020
    assert spec.data[0]["trial_count"] == 2
    assert len(citations) > 0
    assert citations[0].nct_id in ["NCT0001", "NCT0002"]

def test_network_graph_aggregation(sample_studies):
    aggregator = DataAggregator()
    analysis = QueryIntentAnalysis(
        intent="network_sponsors_drugs",
        recommended_visualization=VisualizationType.NETWORK_GRAPH,
        suggested_title="Network Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(sample_studies, analysis, {})

    assert spec.type == VisualizationType.NETWORK_GRAPH
    assert len(spec.nodes) > 0
    assert len(spec.edges) > 0
    node_groups = {n.group for n in spec.nodes}
    assert "sponsor" in node_groups
    assert "drug" in node_groups

def test_network_drug_drug_aggregation():
    aggregator = DataAggregator()
    studies = [
        NormalizedStudy(
            nct_id="NCT0010",
            brief_title="Combo Study of Pembrolizumab and Lenvatinib",
            lead_sponsor="Merck Sharp & Dohme",
            lead_sponsor_class="INDUSTRY",
            interventions=[
                StudyIntervention(name="Pembrolizumab", type="DRUG"),
                StudyIntervention(name="Lenvatinib", type="DRUG"),
            ],
        ),
        NormalizedStudy(
            nct_id="NCT0011",
            brief_title="Single-Agent Pembrolizumab Study",
            lead_sponsor="Merck Sharp & Dohme",
            lead_sponsor_class="INDUSTRY",
            interventions=[StudyIntervention(name="Pembrolizumab", type="DRUG")],
        ),
    ]
    analysis = QueryIntentAnalysis(
        intent="network_drug_drug",
        recommended_visualization=VisualizationType.NETWORK_GRAPH,
        suggested_title="Drug-Drug Network Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(studies, analysis, {})

    node_groups = {n.group for n in spec.nodes}
    assert node_groups == {"drug"}
    assert len(spec.edges) == 1
    assert {spec.edges[0].source, spec.edges[0].target} == {"drug_Pembrolizumab", "drug_Lenvatinib"}
    assert len(citations) > 0


def test_network_condition_sponsor_aggregation(sample_studies):
    aggregator = DataAggregator()
    analysis = QueryIntentAnalysis(
        intent="network_condition_sponsor",
        recommended_visualization=VisualizationType.NETWORK_GRAPH,
        suggested_title="Condition-Sponsor Network Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(sample_studies, analysis, {})

    node_groups = {n.group for n in spec.nodes}
    assert node_groups == {"condition", "sponsor"}
    assert len(spec.edges) > 0
    assert len(citations) > 0


def test_scatter_plot_aggregation(sample_studies):
    aggregator = DataAggregator()
    analysis = QueryIntentAnalysis(
        intent="scatter_enrollment_duration",
        recommended_visualization=VisualizationType.SCATTER_PLOT,
        suggested_title="Scatter Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(sample_studies, analysis, {})

    assert spec.type == VisualizationType.SCATTER_PLOT
    assert len(spec.data) == 3
    assert "duration_months" in spec.data[0]
    assert "enrollment" in spec.data[0]


def test_histogram_aggregation(sample_studies):
    aggregator = DataAggregator()
    analysis = QueryIntentAnalysis(
        intent="enrollment_histogram",
        recommended_visualization=VisualizationType.HISTOGRAM,
        suggested_title="Histogram Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process(sample_studies, analysis, {})

    # sample_studies have enrollments 300, 150, 50 -> "251-500", "101-250", "0-50"
    assert spec.type == VisualizationType.HISTOGRAM
    ranges = {d["enrollment_range"] for d in spec.data}
    assert ranges == {"251-500", "101-250", "0-50"}
    assert sum(d["trial_count"] for d in spec.data) == 3
    assert len(citations) > 0


def test_comparison_aggregation(sample_studies):
    aggregator = DataAggregator()
    # Split the shared fixture into two independently-fetched entity groups,
    # mirroring how the endpoint would run two separate fetch_studies() calls.
    studies_a = [s for s in sample_studies if s.nct_id in ("NCT0001", "NCT0002")]  # Pembrolizumab, phases 3 & 2
    studies_b = [s for s in sample_studies if s.nct_id == "NCT0003"]  # Carboplatin, phase 1

    analysis = QueryIntentAnalysis(
        intent="comparison",
        recommended_visualization=VisualizationType.GROUPED_BAR_CHART,
        suggested_title="Comparison Test",
        query_interpretation="Test"
    )
    spec, meta, citations = aggregator.process_comparison(
        studies_a, "Pembrolizumab", studies_b, "Carboplatin", analysis, {}
    )

    assert spec.type == VisualizationType.GROUPED_BAR_CHART
    series_seen = {d["series"] for d in spec.data}
    assert series_seen == {"Pembrolizumab", "Carboplatin"}
    # Each entity's phase counts must stay isolated to its own series, not merged.
    pembro_phase_3 = next(d for d in spec.data if d["series"] == "Pembrolizumab" and d["phase"] == "Phase 3")
    assert pembro_phase_3["trial_count"] == 1
    carbo_phase_1 = next(d for d in spec.data if d["series"] == "Carboplatin" and d["phase"] == "Phase 1")
    assert carbo_phase_1["trial_count"] == 1
    assert meta.total_trials_analyzed == 3
    assert len(citations) > 0
