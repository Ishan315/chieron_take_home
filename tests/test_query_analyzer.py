import pytest
from app.models.schemas import QueryRequest, VisualizationType
from app.services.query_analyzer import QueryAnalyzerAgent

@pytest.mark.asyncio
async def test_time_trend_query():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(query="How has the number of trials for Pembrolizumab changed over time?")
    analysis = await agent.analyze(req)
    
    assert analysis.intent == "time_trend"
    assert analysis.recommended_visualization == VisualizationType.TIME_SERIES
    assert analysis.search_term.lower() == "pembrolizumab"

@pytest.mark.asyncio
async def test_network_query():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(query="Show a network of sponsors and drugs for Lung Cancer trials.")
    analysis = await agent.analyze(req)
    
    assert analysis.recommended_visualization == VisualizationType.NETWORK_GRAPH
    assert analysis.condition.lower() == "lung cancer"

@pytest.mark.asyncio
async def test_geographic_query():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(query="Which countries have the most recruiting trials for Melanoma?")
    analysis = await agent.analyze(req)
    
    assert analysis.recommended_visualization == VisualizationType.CHOROPLETH_MAP

@pytest.mark.asyncio
async def test_condition_regex_does_not_swallow_trailing_verb():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(query="How has the number of trials for Pembrolizumab changed per year since 2015?")
    analysis = await agent.analyze(req)

    assert analysis.search_term.lower() == "pembrolizumab"
    # "condition" must not swallow the trailing verb ("changed") or be set
    # to the drug name itself, since Pembrolizumab is a drug, not a condition.
    assert analysis.condition is None

@pytest.mark.asyncio
async def test_condition_regex_stops_before_since_clause():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(query="How many trials started each year for Melanoma since 2015?")
    analysis = await agent.analyze(req)

    assert analysis.condition == "Melanoma"

@pytest.mark.asyncio
async def test_explicit_overrides():
    agent = QueryAnalyzerAgent()
    req = QueryRequest(
        query="Generic query",
        drug_name="Keytruda",
        visualization_override=VisualizationType.PIE_CHART
    )
    analysis = await agent.analyze(req)
    
    assert analysis.search_term == "Keytruda"
    assert analysis.recommended_visualization == VisualizationType.PIE_CHART
