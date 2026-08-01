import re
import json
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.schemas import QueryRequest, VisualizationType

logger = logging.getLogger(__name__)

class QueryIntentAnalysis(BaseModel):
    intent: str = Field(
        ...,
        description="Analytical goal: time_trend, phase_distribution, geographic_distribution, network_sponsors_drugs, network_drug_drug, network_condition_sponsor, scatter_enrollment_duration, status_breakdown, sponsor_breakdown, general"
    )
    recommended_visualization: VisualizationType = Field(..., description="Suggested chart type")
    search_term: Optional[str] = Field(None, description="Term for ClinicalTrials query.term")
    condition: Optional[str] = Field(None, description="Disease for ClinicalTrials query.cond")
    sponsor: Optional[str] = Field(None, description="Sponsor for ClinicalTrials query.spons")
    location: Optional[str] = Field(None, description="Location for ClinicalTrials query.locn")
    status: Optional[str] = Field(None, description="Status filter (e.g. RECRUITING)")
    start_year: Optional[int] = Field(None, description="Start year filter")
    end_year: Optional[int] = Field(None, description="End year filter")
    suggested_title: str = Field(..., description="Chart title")
    query_interpretation: str = Field(..., description="Human readable explanation of query interpretation")

class QueryAnalyzerAgent:
    """
    AI-Enabled Query Analyzer Agent using OpenAI GPT-4o-mini structured outputs
    with a Deterministic Rule-Based NLP Fallback Engine.
    """

    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY

    async def analyze(self, request: QueryRequest) -> QueryIntentAnalysis:
        """
        Main entrypoint: Merges explicit request parameters with AI/NLP query interpretation.
        """
        analysis: Optional[QueryIntentAnalysis] = None

        # 1. Try OpenAI if key is configured
        if self.openai_key:
            try:
                analysis = await self._analyze_with_openai(request.query)
            except Exception as e:
                logger.warning(f"OpenAI query analysis failed: {e}. Falling back to rule-based engine...")

        # 2. Fallback to Rule-Based Engine if OpenAI key is missing or call failed
        if not analysis:
            analysis = self._analyze_with_rules(request.query)

        # 3. Override with explicit structured fields if provided by caller
        if request.drug_name:
            analysis.search_term = request.drug_name
        if request.condition:
            analysis.condition = request.condition
        if request.sponsor:
            analysis.sponsor = request.sponsor
        if request.country:
            analysis.location = request.country
        if request.overall_status:
            analysis.status = request.overall_status
        if request.start_year:
            analysis.start_year = request.start_year
        if request.end_year:
            analysis.end_year = request.end_year
        if request.visualization_override:
            analysis.recommended_visualization = request.visualization_override

        return analysis

    async def _analyze_with_openai(self, query: str) -> QueryIntentAnalysis:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.openai_key)
        
        system_prompt = (
            "You are an expert Clinical Trials Data Analyst Agent.\n"
            "Analyze the user's question about clinical trials and map it to ClinicalTrials.gov search parameters and visualization format.\n"
            "Rules for recommended_visualization:\n"
            "- If asking about countries/geography/locations -> choropleth_map\n"
            "- If asking about network/relationships/sponsors to drugs/co-occurrence -> network_graph\n"
            "- If asking about time/trends/years/over time -> time_series\n"
            "- If comparing two things/drugs/sponsors -> grouped_bar_chart\n"
            "- If asking about enrollment vs duration / scatter -> scatter_plot\n"
            "- If asking about status/proportions -> pie_chart or bar_chart\n"
            "- Default for phases/counts -> bar_chart"
        )

        response = await client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format=QueryIntentAnalysis
        )
        return response.choices[0].message.parsed

    def _analyze_with_rules(self, query: str) -> QueryIntentAnalysis:
        q_lower = query.lower()

        # 1. Visual Intent Detection
        viz_type = VisualizationType.BAR_CHART
        intent = "phase_distribution"

        if any(w in q_lower for w in ["network", "relationship", "sponsor to drug", "co-occur", "combination", "connect", "link"]):
            viz_type = VisualizationType.NETWORK_GRAPH
            if "drug to drug" in q_lower or "combination" in q_lower:
                intent = "network_drug_drug"
            elif "condition" in q_lower and "sponsor" in q_lower:
                intent = "network_condition_sponsor"
            else:
                intent = "network_sponsors_drugs"

        elif any(w in q_lower for w in ["over time", "per year", "by year", "trend", "change over", "annual", "history", "since"]):
            viz_type = VisualizationType.TIME_SERIES
            intent = "time_trend"

        elif any(w in q_lower for w in ["country", "countries", "geograph", "location", "world", "where", "nation"]):
            viz_type = VisualizationType.CHOROPLETH_MAP
            intent = "geographic_distribution"

        elif any(w in q_lower for w in ["scatter", "enrollment vs", "duration vs", "size vs", "enrollment and duration"]):
            viz_type = VisualizationType.SCATTER_PLOT
            intent = "scatter_enrollment_duration"

        elif any(w in q_lower for w in ["status", "recruiting vs", "completed vs"]):
            viz_type = VisualizationType.PIE_CHART
            intent = "status_breakdown"

        elif any(w in q_lower for w in ["compare", "vs", "versus"]):
            viz_type = VisualizationType.GROUPED_BAR_CHART
            intent = "phase_distribution"

        elif "phase" in q_lower:
            viz_type = VisualizationType.BAR_CHART
            intent = "phase_distribution"

        # 2. Entity Extraction via Regular Expressions & Keywords
        search_term: Optional[str] = None
        condition: Optional[str] = None
        start_year: Optional[int] = None
        location: Optional[str] = None

        # Extract year
        year_match = re.search(r'\b(20\d\d|19\d\d)\b', query)
        if year_match:
            start_year = int(year_match.group(1))

        # Common drugs check if query mentions specific drug (checked before
        # condition extraction so drug names never get misparsed as conditions)
        drug_keywords = [
            "pembrolizumab", "keytruda", "nivolumab", "opdivo", "rituximab", "trastuzumab",
            "bevacizumab", "atezolizumab", "durvalumab", "semaglutide", "metformin",
            "remdesivir", "paxlovid", "car-t", "aspirin", "adalimumab", "imatinib"
        ]
        for d in drug_keywords:
            if d in q_lower:
                search_term = d.capitalize()
                break

        # Extract condition/disease keywords. The boundary list includes common
        # verbs/connectors (e.g. "changed", "increased") so the capture stops at
        # the end of the noun phrase instead of swallowing the rest of the sentence.
        boundary = (
            r'trials|studies|patients|per|over|since|changed|increased|decreased|grown|risen|'
            r'fallen|shifted|evolved|started|began|has|have|had|is|are|was|were|in\s+[A-Z]'
        )
        condition_patterns = [
            rf'for ([A-Za-z0-9\s\-]+?)(?:\s+(?:{boundary})\b|\?|$)',
            rf'in ([A-Za-z0-9\s\-]+?)(?:\s+(?:{boundary})\b|\?|$)'
        ]
        for pattern in condition_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                is_placeholder = candidate.lower() in ["this drug", "each year", "all countries", "different phases"]
                is_drug = candidate.lower() in drug_keywords or (search_term and candidate.lower() == search_term.lower())
                if not is_placeholder and not is_drug:
                    condition = candidate
                    break

        if not search_term and not condition:
            # Fallback search term from query words excluding generic words
            words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', query) 
                     if w.lower() not in ["how", "many", "trials", "studies", "changed", "over", "time", "show", "what", "where", "which", "number", "distributed", "across", "most", "common"]]
            if words:
                search_term = words[0].capitalize()

        # Build Title
        subject = search_term or condition or "Clinical Trials"
        intent_titles = {
            "time_trend": f"Trial Count Over Time for {subject}",
            "phase_distribution": f"Trial Distribution Across Phases for {subject}",
            "geographic_distribution": f"Geographic Distribution of {subject} Trials",
            "network_sponsors_drugs": f"Sponsor to Drug Network for {subject} Trials",
            "network_drug_drug": f"Drug-Drug Combination Network for {subject}",
            "network_condition_sponsor": f"Condition to Sponsor Network for {subject}",
            "scatter_enrollment_duration": f"Enrollment Count vs. Study Duration for {subject}",
            "status_breakdown": f"Overall Trial Status Breakdown for {subject}"
        }
        title = intent_titles.get(intent, f"Clinical Trials Analysis for {subject}")

        query_interpretation = (
            f"Parsed query as '{intent}' intent focusing on '{subject}'. "
            f"Recommended visualization: {viz_type.value}."
        )

        return QueryIntentAnalysis(
            intent=intent,
            recommended_visualization=viz_type,
            search_term=search_term,
            condition=condition,
            location=location,
            start_year=start_year,
            suggested_title=title,
            query_interpretation=query_interpretation
        )
