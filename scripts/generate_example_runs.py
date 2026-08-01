import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "backend"))

from app.models.schemas import QueryRequest
from app.services.query_analyzer import QueryAnalyzerAgent
from app.services.clinical_trials_client import ClinicalTrialsClient
from app.services.data_aggregator import DataAggregator

EXAMPLES = [
    {
        "filename": "example_1_time_series.json",
        "request": QueryRequest(
            query="How has the number of trials for Pembrolizumab changed over time?",
            drug_name="Pembrolizumab",
            max_trials_to_analyze=100
        )
    },
    {
        "filename": "example_2_phase_distribution.json",
        "request": QueryRequest(
            query="How are Melanoma trials distributed across phases?",
            condition="Melanoma",
            max_trials_to_analyze=100
        )
    },
    {
        "filename": "example_3_network_graph.json",
        "request": QueryRequest(
            query="Show a network of sponsors to drugs for Lung Cancer trials.",
            condition="Lung Cancer",
            visualization_override="network_graph",
            max_trials_to_analyze=100
        )
    },
    {
        "filename": "example_4_geographic.json",
        "request": QueryRequest(
            query="Which countries have the most recruiting trials for Breast Cancer?",
            condition="Breast Cancer",
            overall_status="RECRUITING",
            max_trials_to_analyze=100
        )
    },
    {
        "filename": "example_5_scatter_plot.json",
        "request": QueryRequest(
            query="What is the relationship between trial enrollment count and study duration for Immunotherapy trials?",
            condition="Immunotherapy",
            visualization_override="scatter_plot",
            max_trials_to_analyze=100
        )
    }
]

async def main():
    analyzer = QueryAnalyzerAgent()
    ct_client = ClinicalTrialsClient()
    aggregator = DataAggregator()

    out_dir = root_dir / "examples"
    out_dir.mkdir(exist_ok=True)

    print("Generating example JSON output runs from ClinicalTrials.gov API v2...")

    for item in EXAMPLES:
        req: QueryRequest = item["request"]
        fname = item["filename"]
        out_path = out_dir / fname

        print(f"Processing query: '{req.query}' -> {fname}")
        analysis = await analyzer.analyze(req)
        filters_applied = {
            "condition": analysis.condition,
            "term": analysis.search_term,
            "sponsor": analysis.sponsor,
            "location": analysis.location,
            "status": analysis.status
        }
        filters_applied = {k: v for k, v in filters_applied.items() if v is not None}

        studies = await ct_client.fetch_studies(
            condition=analysis.condition,
            term=analysis.search_term,
            sponsor=analysis.sponsor,
            location=analysis.location,
            status=analysis.status,
            max_results=req.max_trials_to_analyze
        )

        if not studies and analysis.search_term:
            studies = await ct_client.fetch_studies(term=analysis.search_term, max_results=req.max_trials_to_analyze)

        spec, meta, citations = aggregator.process(studies, analysis, filters_applied)

        response_dict = {
            "request_input": req.model_dump(),
            "visualization": spec.model_dump(),
            "meta": meta.model_dump(),
            "citations": [c.model_dump() for c in citations[:15]]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2)

        print(f"Saved {fname} (analyzed {len(studies)} studies, generated {len(citations)} citations)")

    print("\nAll 5 example runs generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
