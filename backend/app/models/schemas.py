from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class VisualizationType(str, Enum):
    BAR_CHART = "bar_chart"
    GROUPED_BAR_CHART = "grouped_bar_chart"
    TIME_SERIES = "time_series"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    CHOROPLETH_MAP = "choropleth_map"
    NETWORK_GRAPH = "network_graph"
    PIE_CHART = "pie_chart"

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language question about clinical trials",
        examples=["How has the number of trials for Pembrolizumab changed over time?"]
    )
    drug_name: Optional[str] = Field(None, description="Target drug / intervention name")
    condition: Optional[str] = Field(None, description="Target condition / disease")
    trial_phase: Optional[str] = Field(None, description="Filter by trial phase (e.g. Phase 1, Phase 3)")
    sponsor: Optional[str] = Field(None, description="Filter by lead sponsor or collaborator")
    country: Optional[str] = Field(None, description="Filter by country location")
    drug_name_b: Optional[str] = Field(None, description="Second drug / intervention name, for A-vs-B comparison queries")
    condition_b: Optional[str] = Field(None, description="Second condition / disease, for A-vs-B comparison queries")
    sponsor_b: Optional[str] = Field(None, description="Second sponsor, for A-vs-B comparison queries")
    start_year: Optional[int] = Field(None, description="Filter studies starting in or after this year")
    end_year: Optional[int] = Field(None, description="Filter studies starting in or before this year")
    overall_status: Optional[str] = Field(None, description="Filter status (e.g., RECRUITING, COMPLETED)")
    visualization_override: Optional[VisualizationType] = Field(None, description="Force a specific visualization format")
    max_trials_to_analyze: int = Field(200, ge=10, le=500, description="Max trial records to retrieve and analyze")

class AxisEncoding(BaseModel):
    field: str
    label: str
    type: str = "nominal"  # nominal, quantitative, temporal, ordinal
    unit: Optional[str] = None

class NetworkEncoding(BaseModel):
    node_id: str = "id"
    node_label: str = "label"
    node_group: str = "group"
    node_val: str = "val"
    edge_source: str = "source"
    edge_target: str = "target"
    edge_weight: str = "weight"

class EncodingSpec(BaseModel):
    x: Optional[AxisEncoding] = None
    y: Optional[AxisEncoding] = None
    group: Optional[AxisEncoding] = None
    color: Optional[AxisEncoding] = None
    size: Optional[AxisEncoding] = None
    network: Optional[NetworkEncoding] = None

class NetworkNode(BaseModel):
    id: str
    label: str
    group: str  # e.g., "sponsor", "drug", "condition"
    val: int = 1  # count/weight for node scaling
    details: Optional[Dict[str, Any]] = None

class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: int = 1
    label: Optional[str] = None

class DeepCitation(BaseModel):
    nct_id: str
    title: str
    excerpt: str
    url: str
    field_name: Optional[str] = None
    data_bucket: Optional[str] = None

class DataPoint(BaseModel):
    # Flexible container for tabular data points (bar charts, time series, scatter, maps)
    # Allows dynamic keys while keeping strict typing where possible
    data: Dict[str, Any]
    citations: List[DeepCitation] = Field(default_factory=list)

class VisualizationSpec(BaseModel):
    type: VisualizationType
    title: str
    subtitle: Optional[str] = None
    encoding: EncodingSpec
    data: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: Optional[List[NetworkNode]] = None
    edges: Optional[List[NetworkEdge]] = None

class MetadataSpec(BaseModel):
    filters_applied: Dict[str, Any]
    source: str = "clinicaltrials.gov"
    total_trials_analyzed: int
    query_interpretation: str
    time_granularity: str = "none"
    units: Dict[str, str] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)

class QueryResponse(BaseModel):
    visualization: VisualizationSpec
    meta: MetadataSpec
    citations: List[DeepCitation] = Field(default_factory=list)
