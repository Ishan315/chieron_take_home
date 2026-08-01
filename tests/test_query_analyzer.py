import pytest
from app.core.config import settings
from app.models.schemas import QueryRequest, VisualizationType
from app.services.query_analyzer import QueryAnalyzerAgent

@pytest.fixture
def agent():
    # These tests exercise the deterministic rule-based fallback specifically
    # (regex entity extraction, keyword-based intent detection), so the OpenAI
    # path is force-disabled regardless of whether OPENAI_API_KEY is set in the
    # environment. Without this, these tests would silently start hitting the
    # live OpenAI API whenever a real key is configured, making them slow,
    # non-deterministic, and billed. The LLM path itself is covered separately.
    a = QueryAnalyzerAgent()
    a.openai_key = None
    return a

@pytest.mark.asyncio
async def test_time_trend_query(agent):
    req = QueryRequest(query="How has the number of trials for Pembrolizumab changed over time?")
    analysis = await agent.analyze(req)

    assert analysis.intent == "time_trend"
    assert analysis.recommended_visualization == VisualizationType.TIME_SERIES
    assert analysis.search_term.lower() == "pembrolizumab"

@pytest.mark.asyncio
async def test_network_query(agent):
    req = QueryRequest(query="Show a network of sponsors and drugs for Lung Cancer trials.")
    analysis = await agent.analyze(req)

    assert analysis.recommended_visualization == VisualizationType.NETWORK_GRAPH
    assert analysis.condition.lower() == "lung cancer"

@pytest.mark.asyncio
async def test_geographic_query(agent):
    req = QueryRequest(query="Which countries have the most recruiting trials for Melanoma?")
    analysis = await agent.analyze(req)

    assert analysis.recommended_visualization == VisualizationType.CHOROPLETH_MAP

@pytest.mark.asyncio
async def test_condition_regex_does_not_swallow_trailing_verb(agent):
    req = QueryRequest(query="How has the number of trials for Pembrolizumab changed per year since 2015?")
    analysis = await agent.analyze(req)

    assert analysis.search_term.lower() == "pembrolizumab"
    # "condition" must not swallow the trailing verb ("changed") or be set
    # to the drug name itself, since Pembrolizumab is a drug, not a condition.
    assert analysis.condition is None

@pytest.mark.asyncio
async def test_condition_regex_stops_before_since_clause(agent):
    req = QueryRequest(query="How many trials started each year for Melanoma since 2015?")
    analysis = await agent.analyze(req)

    assert analysis.condition == "Melanoma"

@pytest.mark.asyncio
async def test_non_latin_script_entity_extraction(agent):
    # The Latin-only word-grab (\b[A-Za-z]{4,}\b) can never match Chinese, so this
    # used to silently fall through to an English filler word ("Distribution")
    # instead of the real condition. Confirmed live that the extracted term
    # actually matches a real diabetes trial on ClinicalTrials.gov.
    req = QueryRequest(query="糖尿病 trials distribution across phases")
    analysis = await agent.analyze(req)

    assert analysis.search_term == "糖尿病"

@pytest.mark.asyncio
async def test_two_drug_comparison_detection(agent):
    req = QueryRequest(query="Compare phases for trials involving pembrolizumab vs nivolumab")
    analysis = await agent.analyze(req)

    assert analysis.intent == "comparison"
    assert analysis.search_term.lower() == "pembrolizumab"
    assert analysis.search_term_b.lower() == "nivolumab"
    assert analysis.recommended_visualization == VisualizationType.GROUPED_BAR_CHART

@pytest.mark.asyncio
async def test_single_drug_query_has_no_comparison_entity(agent):
    req = QueryRequest(query="How has the number of trials for Pembrolizumab changed over time?")
    analysis = await agent.analyze(req)

    assert analysis.search_term_b is None

@pytest.mark.asyncio
async def test_explicit_overrides(agent):
    req = QueryRequest(
        query="Generic query",
        drug_name="Keytruda",
        visualization_override=VisualizationType.PIE_CHART
    )
    analysis = await agent.analyze(req)

    assert analysis.search_term == "Keytruda"
    assert analysis.recommended_visualization == VisualizationType.PIE_CHART

@pytest.mark.asyncio
async def test_explicit_comparison_field_override(agent):
    req = QueryRequest(
        query="Compare sponsor categories across two conditions",
        condition="Melanoma",
        condition_b="Lung Cancer"
    )
    analysis = await agent.analyze(req)

    assert analysis.condition == "Melanoma"
    assert analysis.condition_b == "Lung Cancer"
    # "Compare" must not leak through as a fallback search_term: the caller told
    # us the entities are conditions (not drugs), so a lingering search_term would
    # both mislabel a comparison series and double-filter the live API query.
    assert analysis.search_term is None

@pytest.mark.asyncio
async def test_explicit_condition_without_drug_name_clears_fallback_search_term(agent):
    req = QueryRequest(query="Show me trial data", condition="Melanoma")
    analysis = await agent.analyze(req)

    assert analysis.condition == "Melanoma"
    assert analysis.search_term is None

def test_sanitize_analysis_drops_generic_filler_terms():
    from app.services.query_analyzer import QueryIntentAnalysis
    agent = QueryAnalyzerAgent()
    analysis = QueryIntentAnalysis(
        intent="comparison",
        recommended_visualization=VisualizationType.GROUPED_BAR_CHART,
        search_term="things",  # a generic word invented from underspecified input, not a real drug
        condition_b="Melanoma",
        suggested_title="x",
        query_interpretation="x"
    )
    sanitized = agent._sanitize_analysis(analysis)
    assert sanitized.search_term is None
    assert sanitized.condition_b == "Melanoma"

def test_sanitize_analysis_collapses_redundant_term_and_condition():
    from app.services.query_analyzer import QueryIntentAnalysis
    agent = QueryAnalyzerAgent()
    analysis = QueryIntentAnalysis(
        intent="phase_distribution",
        recommended_visualization=VisualizationType.BAR_CHART,
        search_term="foobar",
        condition="foobar",  # same value in both fields -- redundant, over-constrains the API query
        suggested_title="x",
        query_interpretation="x"
    )
    sanitized = agent._sanitize_analysis(analysis)
    assert sanitized.search_term == "foobar"
    assert sanitized.condition is None

@pytest.mark.asyncio
@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY not configured")
async def test_live_openai_analysis_smoke():
    # Exercises the actual LLM path (structured outputs via response_format=
    # QueryIntentAnalysis), which every other test in this file deliberately
    # bypasses via the `agent` fixture. Only runs when a real key is present.
    live_agent = QueryAnalyzerAgent()
    req = QueryRequest(query="Compare phases for trials involving pembrolizumab vs nivolumab")
    analysis = await live_agent.analyze(req)

    assert analysis.search_term.lower() == "pembrolizumab"
    assert analysis.search_term_b is not None
    assert analysis.search_term_b.lower() == "nivolumab"
    assert analysis.recommended_visualization == VisualizationType.GROUPED_BAR_CHART
