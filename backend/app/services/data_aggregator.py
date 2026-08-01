from collections import defaultdict, Counter
from itertools import combinations
from typing import List, Dict, Any, Tuple
from app.models.clinical_trials import NormalizedStudy
from app.models.schemas import (
    VisualizationType, VisualizationSpec, EncodingSpec, AxisEncoding,
    NetworkEncoding, NetworkNode, NetworkEdge, DeepCitation, MetadataSpec
)
from app.services.query_analyzer import QueryIntentAnalysis

class DataAggregator:
    """
    Transforms normalized clinical trial studies into structured visualization specifications
    complete with encodings, chart-ready data, network nodes/edges, and deep citations.
    """

    def process(
        self,
        studies: List[NormalizedStudy],
        analysis: QueryIntentAnalysis,
        filters_applied: Dict[str, Any]
    ) -> Tuple[VisualizationSpec, MetadataSpec, List[DeepCitation]]:
        
        viz_type = analysis.recommended_visualization
        all_citations: List[DeepCitation] = []

        if viz_type == VisualizationType.TIME_SERIES:
            spec, citations = self._aggregate_time_series(studies, analysis)
        elif viz_type == VisualizationType.CHOROPLETH_MAP:
            spec, citations = self._aggregate_geographic(studies, analysis)
        elif viz_type == VisualizationType.NETWORK_GRAPH:
            if analysis.intent == "network_drug_drug":
                spec, citations = self._aggregate_network_drug_drug(studies, analysis)
            elif analysis.intent == "network_condition_sponsor":
                spec, citations = self._aggregate_network_condition_sponsor(studies, analysis)
            else:
                spec, citations = self._aggregate_network_graph(studies, analysis)
        elif viz_type == VisualizationType.SCATTER_PLOT:
            spec, citations = self._aggregate_scatter_plot(studies, analysis)
        elif viz_type == VisualizationType.GROUPED_BAR_CHART:
            spec, citations = self._aggregate_grouped_bar_chart(studies, analysis)
        elif viz_type == VisualizationType.PIE_CHART:
            spec, citations = self._aggregate_pie_chart(studies, analysis)
        elif viz_type == VisualizationType.HISTOGRAM:
            spec, citations = self._aggregate_histogram(studies, analysis)
        else:  # Default BAR_CHART
            spec, citations = self._aggregate_bar_chart(studies, analysis)

        all_citations.extend(citations)

        meta = MetadataSpec(
            filters_applied=filters_applied,
            source="clinicaltrials.gov",
            total_trials_analyzed=len(studies),
            query_interpretation=analysis.query_interpretation,
            time_granularity="year" if viz_type == VisualizationType.TIME_SERIES else "none",
            units={"y": "trial_count"},
            notes=[
                f"Aggregated {len(studies)} clinical trial records from ClinicalTrials.gov API v2.",
                f"Generated {len(all_citations)} deep source citations linking back to individual study NCT IDs."
            ]
        )

        return spec, meta, all_citations

    def _make_citation(self, study: NormalizedStudy, field_name: str, excerpt: str, bucket: str) -> DeepCitation:
        return DeepCitation(
            nct_id=study.nct_id,
            title=study.brief_title,
            excerpt=excerpt[:250] + ("..." if len(excerpt) > 250 else ""),
            url=f"https://clinicaltrials.gov/study/{study.nct_id}",
            field_name=field_name,
            data_bucket=bucket
        )

    def _aggregate_time_series(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        year_counts: Dict[int, int] = defaultdict(int)
        year_studies: Dict[int, List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            if s.start_year and 1990 <= s.start_year <= 2030:
                year_counts[s.start_year] += 1
                year_studies[s.start_year].append(s)

        sorted_years = sorted(year_counts.keys())
        data_points = []
        citations = []

        for yr in sorted_years:
            bucket_studies = year_studies[yr]
            sample_citations = [
                self._make_citation(
                    st,
                    "startDateStruct",
                    f"Study started in {st.start_year}: '{st.official_title or st.brief_title}' ({st.nct_id})",
                    str(yr)
                ) for st in bucket_studies[:3]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "year": yr,
                "trial_count": year_counts[yr],
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.TIME_SERIES,
            title=analysis.suggested_title,
            subtitle="Annual count of clinical trial starts",
            encoding=EncodingSpec(
                x=AxisEncoding(field="year", label="Year", type="temporal"),
                y=AxisEncoding(field="trial_count", label="Number of Trials Started", type="quantitative", unit="trials")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_bar_chart(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        phase_counts: Dict[str, int] = defaultdict(int)
        phase_studies: Dict[str, List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            phases = s.phases if s.phases else ["Not Applicable"]
            for p in phases:
                phase_counts[p] += 1
                phase_studies[p].append(s)

        # Standard phase ordering
        preferred_order = ["Early Phase 1", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"]
        sorted_phases = sorted(phase_counts.keys(), key=lambda x: preferred_order.index(x) if x in preferred_order else 99)

        data_points = []
        citations = []

        for p in sorted_phases:
            bucket_studies = phase_studies[p]
            sample_citations = [
                self._make_citation(
                    st,
                    "designModule.phases",
                    f"{p} study: '{st.brief_title}' (Sponsor: {st.lead_sponsor})",
                    p
                ) for st in bucket_studies[:3]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "phase": p,
                "trial_count": phase_counts[p],
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.BAR_CHART,
            title=analysis.suggested_title,
            subtitle="Distribution of clinical trials across phases",
            encoding=EncodingSpec(
                x=AxisEncoding(field="phase", label="Trial Phase", type="nominal"),
                y=AxisEncoding(field="trial_count", label="Number of Trials", type="quantitative", unit="trials")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_grouped_bar_chart(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        grouped_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        grouped_studies: Dict[Tuple[str, str], List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            spon_class = s.lead_sponsor_class or "OTHER"
            spon_label = "Industry" if spon_class == "INDUSTRY" else ("NIH/Gov" if spon_class in ["NIH", "FED"] else "Academic/Other")
            for p in (s.phases or ["Not Applicable"]):
                grouped_counts[(p, spon_label)] += 1
                grouped_studies[(p, spon_label)].append(s)

        data_points = []
        citations = []

        for (p, spon_label), count in grouped_counts.items():
            bucket_studies = grouped_studies[(p, spon_label)]
            sample_citations = [
                self._make_citation(
                    st,
                    "sponsorCollaboratorsModule",
                    f"{spon_label} sponsored {p} trial: '{st.brief_title}'",
                    f"{p} - {spon_label}"
                ) for st in bucket_studies[:2]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "phase": p,
                "sponsor_group": spon_label,
                "trial_count": count,
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.GROUPED_BAR_CHART,
            title=analysis.suggested_title,
            subtitle="Comparison of phases broken down by sponsor category",
            encoding=EncodingSpec(
                x=AxisEncoding(field="phase", label="Trial Phase", type="nominal"),
                y=AxisEncoding(field="trial_count", label="Number of Trials", type="quantitative", unit="trials"),
                group=AxisEncoding(field="sponsor_group", label="Sponsor Category", type="nominal")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_geographic(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        country_counts: Dict[str, int] = defaultdict(int)
        country_studies: Dict[str, List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            for c in s.countries:
                country_counts[c] += 1
                country_studies[c].append(s)

        sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:25]
        data_points = []
        citations = []

        for country, count in sorted_countries:
            bucket_studies = country_studies[country]
            sample_citations = [
                self._make_citation(
                    st,
                    "contactsLocationsModule",
                    f"Trial active in {country}: '{st.brief_title}' ({st.nct_id})",
                    country
                ) for st in bucket_studies[:3]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "country": country,
                "trial_count": count,
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.CHOROPLETH_MAP,
            title=analysis.suggested_title,
            subtitle="Count of trials by country location",
            encoding=EncodingSpec(
                x=AxisEncoding(field="country", label="Country", type="nominal"),
                y=AxisEncoding(field="trial_count", label="Trial Count", type="quantitative", unit="trials")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_scatter_plot(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        data_points = []
        citations = []

        for s in studies:
            if s.enrollment and s.start_year and s.completion_year and s.completion_year >= s.start_year:
                duration_months = max(1, (s.completion_year - s.start_year) * 12)
                cit = self._make_citation(
                    s,
                    "designModule.enrollmentInfo",
                    f"Enrollment: {s.enrollment} patients, Duration: ~{duration_months} months for study '{s.brief_title}'",
                    s.nct_id
                )
                citations.append(cit)

                data_points.append({
                    "nct_id": s.nct_id,
                    "title": s.brief_title,
                    "enrollment": s.enrollment,
                    "duration_months": duration_months,
                    "phase": s.phases[0] if s.phases else "N/A",
                    "lead_sponsor": s.lead_sponsor,
                    "citations": [cit.model_dump()]
                })

        spec = VisualizationSpec(
            type=VisualizationType.SCATTER_PLOT,
            title=analysis.suggested_title,
            subtitle="Study enrollment size versus estimated duration in months",
            encoding=EncodingSpec(
                x=AxisEncoding(field="duration_months", label="Estimated Duration (Months)", type="quantitative", unit="months"),
                y=AxisEncoding(field="enrollment", label="Enrollment Count (Patients)", type="quantitative", unit="patients"),
                size=AxisEncoding(field="enrollment", label="Enrollment Size", type="quantitative"),
                color=AxisEncoding(field="phase", label="Phase", type="nominal")
            ),
            data=data_points[:100]  # Cap scatter points for responsive rendering
        )
        return spec, citations

    def _aggregate_histogram(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        """
        Builds a histogram of trial enrollment sizes, binned into fixed patient-count ranges.
        """
        bin_edges = [(0, 50), (51, 100), (101, 250), (251, 500), (501, 1000), (1001, 5000), (5001, float("inf"))]
        bin_labels = ["0-50", "51-100", "101-250", "251-500", "501-1000", "1001-5000", "5000+"]

        bucket_counts: Dict[str, int] = defaultdict(int)
        bucket_studies: Dict[str, List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            if not s.enrollment or s.enrollment <= 0:
                continue
            for (low, high), label in zip(bin_edges, bin_labels):
                if low <= s.enrollment <= high:
                    bucket_counts[label] += 1
                    bucket_studies[label].append(s)
                    break

        data_points = []
        citations = []
        for label in bin_labels:
            if label not in bucket_counts:
                continue
            b_studies = bucket_studies[label]
            sample_citations = [
                self._make_citation(
                    st,
                    "designModule.enrollmentInfo",
                    f"Enrollment of {st.enrollment} patients: '{st.brief_title}' ({st.nct_id})",
                    label
                ) for st in b_studies[:3]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "enrollment_range": label,
                "trial_count": bucket_counts[label],
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.HISTOGRAM,
            title=analysis.suggested_title,
            subtitle="Distribution of trial enrollment sizes",
            encoding=EncodingSpec(
                x=AxisEncoding(field="enrollment_range", label="Enrollment Size Range (Patients)", type="ordinal"),
                y=AxisEncoding(field="trial_count", label="Number of Trials", type="quantitative", unit="trials")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_pie_chart(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        status_counts = Counter(s.overall_status for s in studies)
        status_studies: Dict[str, List[NormalizedStudy]] = defaultdict(list)
        for s in studies:
            status_studies[s.overall_status].append(s)

        data_points = []
        citations = []

        for status, count in status_counts.most_common():
            bucket_studies = status_studies[status]
            sample_citations = [
                self._make_citation(
                    st,
                    "statusModule.overallStatus",
                    f"Status '{status}': '{st.brief_title}'",
                    status
                ) for st in bucket_studies[:3]
            ]
            citations.extend(sample_citations)

            data_points.append({
                "status": status,
                "trial_count": count,
                "citations": [c.model_dump() for c in sample_citations]
            })

        spec = VisualizationSpec(
            type=VisualizationType.PIE_CHART,
            title=analysis.suggested_title,
            subtitle="Overall status breakdown of trials",
            encoding=EncodingSpec(
                x=AxisEncoding(field="status", label="Status", type="nominal"),
                y=AxisEncoding(field="trial_count", label="Trial Count", type="quantitative", unit="trials")
            ),
            data=data_points
        )
        return spec, citations

    def _aggregate_network_graph(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        """
        Builds a rich Bipartite / Co-occurrence Network Graph:
        - Nodes: Sponsors (group='sponsor') and Interventions/Drugs (group='drug')
        - Edges: Connection when a sponsor conducts a trial featuring the drug
        """
        sponsor_counts: Dict[str, int] = defaultdict(int)
        drug_counts: Dict[str, int] = defaultdict(int)
        edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
        edge_studies: Dict[Tuple[str, str], List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            sp = s.lead_sponsor
            if sp and sp != "Unknown Sponsor":
                sponsor_counts[sp] += 1
                for inter in s.interventions:
                    d_name = inter.name.strip().title()
                    if len(d_name) > 2 and d_name.lower() not in ["placebo", "other"]:
                        drug_counts[d_name] += 1
                        pair = (f"spon_{sp}", f"drug_{d_name}")
                        edge_weights[pair] += 1
                        edge_studies[pair].append(s)

        # Filter top 15 sponsors and top 15 drugs to keep graph legible
        top_sponsors = set(dict(Counter(sponsor_counts).most_common(15)).keys())
        top_drugs = set(dict(Counter(drug_counts).most_common(15)).keys())

        nodes: List[NetworkNode] = []
        edges: List[NetworkEdge] = []
        citations: List[DeepCitation] = []

        # Add Sponsor Nodes
        for sp in top_sponsors:
            nodes.append(NetworkNode(
                id=f"spon_{sp}",
                label=sp,
                group="sponsor",
                val=sponsor_counts[sp]
            ))

        # Add Drug Nodes
        for dr in top_drugs:
            nodes.append(NetworkNode(
                id=f"drug_{dr}",
                label=dr,
                group="drug",
                val=drug_counts[dr]
            ))

        # Add Edges with Citations
        for (src, tgt), weight in edge_weights.items():
            sp_name = src.replace("spon_", "")
            dr_name = tgt.replace("drug_", "")
            if sp_name in top_sponsors and dr_name in top_drugs:
                b_studies = edge_studies[(src, tgt)]
                sample_cit = [
                    self._make_citation(
                        st,
                        "armsInterventionsModule",
                        f"Sponsor '{sp_name}' evaluated drug '{dr_name}' in study '{st.brief_title}' ({st.nct_id})",
                        f"{sp_name} <-> {dr_name}"
                    ) for st in b_studies[:2]
                ]
                citations.extend(sample_cit)

                edges.append(NetworkEdge(
                    source=src,
                    target=tgt,
                    weight=weight,
                    label=f"{weight} trials"
                ))

        spec = VisualizationSpec(
            type=VisualizationType.NETWORK_GRAPH,
            title=analysis.suggested_title,
            subtitle="Network graph mapping lead sponsors to clinical interventions",
            encoding=EncodingSpec(
                network=NetworkEncoding(
                    node_id="id",
                    node_label="label",
                    node_group="group",
                    edge_source="source",
                    edge_target="target",
                    edge_weight="weight"
                )
            ),
            data=[],
            nodes=nodes,
            edges=edges
        )
        return spec, citations

    def _aggregate_network_drug_drug(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        """
        Builds a Drug-Drug Co-occurrence Network Graph:
        - Nodes: Drugs/Interventions (group='drug')
        - Edges: Connection when two drugs are evaluated together in the same combination study
        """
        drug_counts: Dict[str, int] = defaultdict(int)
        edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
        edge_studies: Dict[Tuple[str, str], List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            drug_names = sorted({
                inter.name.strip().title() for inter in s.interventions
                if len(inter.name.strip()) > 2 and inter.name.strip().lower() not in ["placebo", "other"]
            })
            for d in drug_names:
                drug_counts[d] += 1
            for d1, d2 in combinations(drug_names, 2):
                pair = tuple(sorted((d1, d2)))
                edge_weights[pair] += 1
                edge_studies[pair].append(s)

        top_drugs = set(dict(Counter(drug_counts).most_common(20)).keys())

        nodes: List[NetworkNode] = [
            NetworkNode(id=f"drug_{d}", label=d, group="drug", val=drug_counts[d])
            for d in top_drugs
        ]

        edges: List[NetworkEdge] = []
        citations: List[DeepCitation] = []
        for (d1, d2), weight in edge_weights.items():
            if d1 in top_drugs and d2 in top_drugs:
                b_studies = edge_studies[(d1, d2)]
                sample_cit = [
                    self._make_citation(
                        st,
                        "armsInterventionsModule",
                        f"Combination study evaluating '{d1}' and '{d2}' together: '{st.brief_title}' ({st.nct_id})",
                        f"{d1} <-> {d2}"
                    ) for st in b_studies[:2]
                ]
                citations.extend(sample_cit)
                edges.append(NetworkEdge(
                    source=f"drug_{d1}",
                    target=f"drug_{d2}",
                    weight=weight,
                    label=f"{weight} co-occurring trials"
                ))

        spec = VisualizationSpec(
            type=VisualizationType.NETWORK_GRAPH,
            title=analysis.suggested_title,
            subtitle="Network graph of drugs that co-occur in combination studies",
            encoding=EncodingSpec(
                network=NetworkEncoding(
                    node_id="id",
                    node_label="label",
                    node_group="group",
                    edge_source="source",
                    edge_target="target",
                    edge_weight="weight"
                )
            ),
            data=[],
            nodes=nodes,
            edges=edges
        )
        return spec, citations

    def _aggregate_network_condition_sponsor(self, studies: List[NormalizedStudy], analysis: QueryIntentAnalysis) -> Tuple[VisualizationSpec, List[DeepCitation]]:
        """
        Builds a Condition-Sponsor Network Graph:
        - Nodes: Conditions (group='condition') and Lead Sponsors (group='sponsor')
        - Edges: Connection when a sponsor runs a trial studying a given condition
        """
        condition_counts: Dict[str, int] = defaultdict(int)
        sponsor_counts: Dict[str, int] = defaultdict(int)
        edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
        edge_studies: Dict[Tuple[str, str], List[NormalizedStudy]] = defaultdict(list)

        for s in studies:
            sp = s.lead_sponsor
            if not sp or sp == "Unknown Sponsor":
                continue
            sponsor_counts[sp] += 1
            for c in s.conditions:
                c_name = c.strip()
                if len(c_name) > 2:
                    condition_counts[c_name] += 1
                    pair = (f"cond_{c_name}", f"spon_{sp}")
                    edge_weights[pair] += 1
                    edge_studies[pair].append(s)

        top_conditions = set(dict(Counter(condition_counts).most_common(15)).keys())
        top_sponsors = set(dict(Counter(sponsor_counts).most_common(15)).keys())

        nodes: List[NetworkNode] = [
            NetworkNode(id=f"cond_{c}", label=c, group="condition", val=condition_counts[c])
            for c in top_conditions
        ] + [
            NetworkNode(id=f"spon_{sp}", label=sp, group="sponsor", val=sponsor_counts[sp])
            for sp in top_sponsors
        ]

        edges: List[NetworkEdge] = []
        citations: List[DeepCitation] = []
        for (src, tgt), weight in edge_weights.items():
            c_name = src.replace("cond_", "")
            sp_name = tgt.replace("spon_", "")
            if c_name in top_conditions and sp_name in top_sponsors:
                b_studies = edge_studies[(src, tgt)]
                sample_cit = [
                    self._make_citation(
                        st,
                        "conditionsModule",
                        f"Sponsor '{sp_name}' ran a '{c_name}' trial: '{st.brief_title}' ({st.nct_id})",
                        f"{c_name} <-> {sp_name}"
                    ) for st in b_studies[:2]
                ]
                citations.extend(sample_cit)
                edges.append(NetworkEdge(
                    source=src,
                    target=tgt,
                    weight=weight,
                    label=f"{weight} trials"
                ))

        spec = VisualizationSpec(
            type=VisualizationType.NETWORK_GRAPH,
            title=analysis.suggested_title,
            subtitle="Network graph mapping conditions to lead sponsors",
            encoding=EncodingSpec(
                network=NetworkEncoding(
                    node_id="id",
                    node_label="label",
                    node_group="group",
                    edge_source="source",
                    edge_target="target",
                    edge_weight="weight"
                )
            ),
            data=[],
            nodes=nodes,
            edges=edges
        )
        return spec, citations
