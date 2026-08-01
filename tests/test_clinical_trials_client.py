import pytest
from app.services.clinical_trials_client import ClinicalTrialsClient

def test_extract_year():
    client = ClinicalTrialsClient()
    assert client._extract_year("2021-05-15") == 2021
    assert client._extract_year("May 2018") == 2018
    assert client._extract_year("2015") == 2015
    assert client._extract_year(None) is None
    assert client._extract_year("Invalid") is None

def test_normalize_phase():
    client = ClinicalTrialsClient()
    assert client._normalize_phase("PHASE1") == "Phase 1"
    assert client._normalize_phase("EARLY_PHASE1") == "Early Phase 1"
    assert client._normalize_phase("PHASE3") == "Phase 3"
    assert client._normalize_phase("NA") == "Not Applicable"

def test_parse_study_item():
    client = ClinicalTrialsClient()
    sample_raw = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT01234567",
                "briefTitle": "Test Pembrolizumab Study",
                "officialTitle": "A Phase 3 Study of Pembrolizumab"
            },
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2020-01-15"}
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Merck Sharp & Dohme LLC", "class": "INDUSTRY"}
            },
            "conditionsModule": {
                "conditions": ["Melanoma"]
            },
            "armsInterventionsModule": {
                "interventions": [{"name": "Pembrolizumab", "type": "DRUG"}]
            },
            "designModule": {
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 250}
            },
            "contactsLocationsModule": {
                "locations": [{"country": "United States"}, {"country": "Canada"}]
            }
        }
    }
    study = client._parse_study_item(sample_raw)
    assert study.nct_id == "NCT01234567"
    assert study.brief_title == "Test Pembrolizumab Study"
    assert study.start_year == 2020
    assert study.lead_sponsor == "Merck Sharp & Dohme LLC"
    assert study.phases == ["Phase 3"]
    assert study.interventions[0].name == "Pembrolizumab"
    assert study.countries == ["Canada", "United States"]
    assert study.enrollment == 250

@pytest.mark.asyncio
async def test_live_fetch_studies_smoke():
    client = ClinicalTrialsClient()
    # Live fetch 5 studies for Pembrolizumab to test real ClinicalTrials.gov API v2 connection
    studies = await client.fetch_studies(term="Pembrolizumab", max_results=5)
    assert isinstance(studies, list)
    if len(studies) > 0:
        s = studies[0]
        assert s.nct_id.startswith("NCT")
        assert len(s.brief_title) > 0

@pytest.mark.asyncio
async def test_live_fetch_studies_respects_year_range():
    client = ClinicalTrialsClient()
    studies = await client.fetch_studies(
        term="Pembrolizumab", start_year=2020, end_year=2021, max_results=20
    )
    assert len(studies) > 0
    for s in studies:
        assert s.start_year is not None
        assert 2020 <= s.start_year <= 2021
