import asyncio
import json
import sys
from pathlib import Path

# Add backend directory to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "backend"))

from app.models.schemas import QueryRequest
from app.api.v1.endpoints.query import process_query

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
    },
    {
        "filename": "example_6_drug_drug_network.json",
        "request": QueryRequest(
            query="Which drugs frequently co-occur in combination studies for Lung Cancer?",
            condition="Lung Cancer",
            max_trials_to_analyze=100
        )
    },
    {
        "filename": "example_7_time_series_year_range.json",
        "request": QueryRequest(
            query="How has the number of trials for Pembrolizumab changed per year since 2018?",
            drug_name="Pembrolizumab",
            start_year=2018,
            end_year=2023,
            max_trials_to_analyze=200
        )
    },
    {
        "filename": "example_8_drug_comparison.json",
        "request": QueryRequest(
            query="Compare phases for trials involving Pembrolizumab vs Nivolumab",
            max_trials_to_analyze=200
        )
    }
]

async def main():
    out_dir = root_dir / "examples"
    out_dir.mkdir(exist_ok=True)

    print("Generating example JSON output runs from ClinicalTrials.gov API v2...")

    for item in EXAMPLES:
        req: QueryRequest = item["request"]
        fname = item["filename"]
        out_path = out_dir / fname

        print(f"Processing query: '{req.query}' -> {fname}")

        # Calls the same process_query() the live /api/v1/query endpoint uses,
        # so these example outputs can never drift from real request behavior.
        response = await process_query(req)

        response_dict = {
            "request_input": req.model_dump(),
            "visualization": response.visualization.model_dump(),
            "meta": response.meta.model_dump(),
            "citations": [c.model_dump() for c in response.citations[:15]]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2)

        print(f"Saved {fname} (analyzed {response.meta.total_trials_analyzed} studies, "
              f"generated {len(response.citations)} citations)")

    print(f"\nAll {len(EXAMPLES)} example runs generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
