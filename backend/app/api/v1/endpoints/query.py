from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, VisualizationType
from app.services.query_analyzer import QueryAnalyzerAgent
from app.services.clinical_trials_client import ClinicalTrialsClient
from app.services.data_aggregator import DataAggregator

router = APIRouter()

analyzer_agent = QueryAnalyzerAgent()
ct_client = ClinicalTrialsClient()
aggregator = DataAggregator()

PRESET_EXAMPLES = [
    {
        "id": "time_series_pembrolizumab",
        "title": "Pembrolizumab Time Trend",
        "request": {
            "query": "How has the number of trials for Pembrolizumab changed over time?",
            "drug_name": "Pembrolizumab"
        }
    },
    {
        "id": "phase_distribution_melanoma",
        "title": "Melanoma Phase Distribution",
        "request": {
            "query": "How are Melanoma trials distributed across phases?",
            "condition": "Melanoma"
        }
    },
    {
        "id": "network_sponsor_drug",
        "title": "Sponsor-Drug Network Graph",
        "request": {
            "query": "Show a network of sponsors to drugs for Lung Cancer trials.",
            "condition": "Lung Cancer",
            "visualization_override": "network_graph"
        }
    },
    {
        "id": "geographic_recruiting",
        "title": "Geographic Distribution of Recruiting Trials",
        "request": {
            "query": "Which countries have the most recruiting trials for Breast Cancer?",
            "condition": "Breast Cancer",
            "overall_status": "RECRUITING"
        }
    },
    {
        "id": "scatter_enrollment_duration",
        "title": "Enrollment vs. Duration Scatter Plot",
        "request": {
            "query": "What is the relationship between trial enrollment count and study duration for Immunotherapy trials?",
            "query_term": "Immunotherapy",
            "visualization_override": "scatter_plot"
        }
    }
]

@router.post("/query", response_model=QueryResponse, summary="Convert Clinical Trial Query to Visualization Specification")
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main AI Agent Endpoint:
    1. Interprets natural language question & input parameters
    2. Retrieves real-time trial data from ClinicalTrials.gov API v2
    3. Builds structured visualization spec with encodings, chart data / network graph
    4. Generates deep citations tracing back to exact NCT IDs and field excerpts
    """
    try:
        # Step 1: AI / Rule-Based Query Analysis
        analysis = await analyzer_agent.analyze(request)

        # Step 2: Formulate ClinicalTrials.gov API parameters
        filters_applied = {
            "condition": analysis.condition,
            "term": analysis.search_term,
            "sponsor": analysis.sponsor,
            "condition_b": analysis.condition_b,
            "term_b": analysis.search_term_b,
            "sponsor_b": analysis.sponsor_b,
            "location": analysis.location,
            "status": analysis.status,
            "start_year": analysis.start_year,
            "end_year": analysis.end_year
        }
        # Clean null values
        filters_applied = {k: v for k, v in filters_applied.items() if v is not None}

        is_comparison = bool(analysis.search_term_b or analysis.condition_b or analysis.sponsor_b)

        # A comparison where both sides resolve to the same entity (e.g. "compare
        # Pembrolizumab vs Pembrolizumab") would fetch the same trials twice and
        # emit two data points with an identical (phase, series) key per phase --
        # indistinguishable to any standard grouped-bar renderer. Collapse it to a
        # normal single-entity query instead of returning that duplicate output.
        collapsed_self_comparison = False
        if is_comparison:
            same_entity = (
                (analysis.search_term and analysis.search_term_b
                 and analysis.search_term.strip().lower() == analysis.search_term_b.strip().lower())
                or (analysis.condition and analysis.condition_b
                    and analysis.condition.strip().lower() == analysis.condition_b.strip().lower())
                or (analysis.sponsor and analysis.sponsor_b
                    and analysis.sponsor.strip().lower() == analysis.sponsor_b.strip().lower())
            )
            if same_entity:
                is_comparison = False
                collapsed_self_comparison = True
                analysis.search_term_b = None
                analysis.condition_b = None
                analysis.sponsor_b = None
                analysis.recommended_visualization = VisualizationType.BAR_CHART
                filters_applied.pop("term_b", None)
                filters_applied.pop("condition_b", None)
                filters_applied.pop("sponsor_b", None)

        if is_comparison:
            # Step 3 (comparison path): independently fetch each entity's trials so
            # neither one dilutes or shadows the other in a single merged result set.
            label_a = analysis.search_term or analysis.condition or analysis.sponsor or "Group A"
            label_b = analysis.search_term_b or analysis.condition_b or analysis.sponsor_b or "Group B"

            studies_a = await ct_client.fetch_studies(
                condition=analysis.condition,
                term=analysis.search_term,
                sponsor=analysis.sponsor,
                location=analysis.location,
                status=analysis.status,
                start_year=analysis.start_year,
                end_year=analysis.end_year,
                max_results=request.max_trials_to_analyze
            )
            studies_b = await ct_client.fetch_studies(
                condition=analysis.condition_b,
                term=analysis.search_term_b,
                sponsor=analysis.sponsor_b,
                location=analysis.location,
                status=analysis.status,
                start_year=analysis.start_year,
                end_year=analysis.end_year,
                max_results=request.max_trials_to_analyze
            )

            # Step 4 (comparison path): build the grouped comparison spec + citations
            spec, meta, citations = aggregator.process_comparison(
                studies_a, label_a, studies_b, label_b, analysis, filters_applied
            )
        else:
            # Step 3: Fetch Studies from ClinicalTrials.gov API
            studies = await ct_client.fetch_studies(
                condition=analysis.condition,
                term=analysis.search_term,
                sponsor=analysis.sponsor,
                location=analysis.location,
                status=analysis.status,
                start_year=analysis.start_year,
                end_year=analysis.end_year,
                max_results=request.max_trials_to_analyze
            )

            if not studies:
                # Retry with broader search term if specific condition produced 0 results
                if analysis.search_term:
                    studies = await ct_client.fetch_studies(term=analysis.search_term, max_results=request.max_trials_to_analyze)

            # Step 4: Data Aggregation & Deep Citation Extraction
            spec, meta, citations = aggregator.process(studies, analysis, filters_applied)

            if collapsed_self_comparison:
                meta.notes.append(
                    "Both sides of the requested comparison referred to the same entity; "
                    "showing a single distribution instead of a duplicate comparison."
                )

        return QueryResponse(
            visualization=spec,
            meta=meta,
            citations=citations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing clinical trials query: {str(e)}")

@router.get("/examples", summary="Get Preset Example Queries")
async def get_examples() -> List[Dict[str, Any]]:
    return PRESET_EXAMPLES
