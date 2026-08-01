import pytest
from pydantic import ValidationError
from app.models.schemas import (
    QueryRequest, QueryResponse, VisualizationSpec, VisualizationType,
    EncodingSpec, AxisEncoding, MetadataSpec, DeepCitation
)

def test_query_request_defaults():
    req = QueryRequest(query="How many trials for Melanoma?")
    assert req.query == "How many trials for Melanoma?"
    assert req.max_trials_to_analyze == 200
    assert req.drug_name is None

@pytest.mark.parametrize("start_year,end_year", [
    (3000, None),   # beyond the ClinicalTrials.gov API's own date-parser range (confirmed via live testing)
    (-100, None),   # negative years are rejected by the upstream API's date parser
    (2024, 2015),   # inverted range
])
def test_query_request_rejects_invalid_year_ranges(start_year, end_year):
    with pytest.raises(ValidationError):
        QueryRequest(query="Pembrolizumab trials", start_year=start_year, end_year=end_year)

def test_query_request_accepts_valid_year_range():
    req = QueryRequest(query="Pembrolizumab trials", start_year=2015, end_year=2024)
    assert req.start_year == 2015
    assert req.end_year == 2024

def test_visualization_spec_bar_chart():
    spec = VisualizationSpec(
        type=VisualizationType.BAR_CHART,
        title="Trial count by phase",
        encoding=EncodingSpec(
            x=AxisEncoding(field="phase", label="Trial Phase", type="nominal"),
            y=AxisEncoding(field="trial_count", label="Number of Trials", type="quantitative", unit="trials")
        ),
        data=[
            {"phase": "Phase 1", "trial_count": 32},
            {"phase": "Phase 2", "trial_count": 78}
        ]
    )
    assert spec.type == VisualizationType.BAR_CHART
    assert len(spec.data) == 2
    assert spec.encoding.x.field == "phase"

def test_query_response_serialization():
    citation = DeepCitation(
        nct_id="NCT01234567",
        title="Trial Title",
        excerpt="Phase 3 randomized study evaluating pembrolizumab...",
        url="https://clinicaltrials.gov/study/NCT01234567"
    )
    resp = QueryResponse(
        visualization=VisualizationSpec(
            type=VisualizationType.TIME_SERIES,
            title="Trials over time",
            encoding=EncodingSpec(
                x=AxisEncoding(field="year", label="Year", type="temporal"),
                y=AxisEncoding(field="trial_count", label="Trial Count", type="quantitative")
            ),
            data=[{"year": 2020, "trial_count": 12}]
        ),
        meta=MetadataSpec(
            filters_applied={"drug_name": "Pembrolizumab"},
            total_trials_analyzed=100,
            query_interpretation="Analyzed trial start dates by year for Pembrolizumab"
        ),
        citations=[citation]
    )
    json_data = resp.model_dump()
    assert json_data["visualization"]["type"] == "time_series"
    assert json_data["citations"][0]["nct_id"] == "NCT01234567"
