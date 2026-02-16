"""Lit Spider — automated literature search for simulator parameter validation.

Queries PubMed for each simulator parameter, retrieves abstracts, uses a local
LLM (Ollama) to extract numerical values, and compiles a report comparing
literature values to the simulation's current values.

Pipeline:
    Parameter Registry → PubMed Search → Abstract Retrieval → LLM Extraction → Report

Usage:
    python lit_spider.py                                     # full run (all params, LLM)
    python lit_spider.py --params heteroplasmy_cliff,base_replication_rate
    python lit_spider.py --no-llm                            # keyword-only extraction
    python lit_spider.py --max-papers 5                      # fewer papers per param
    python lit_spider.py --no-cache                          # skip PubMed cache

Requires:
    - Internet access (PubMed E-utilities)
    - Ollama running at localhost:11434 (optional, falls back to keyword extraction)

Output:
    artifacts/lit_spider.json          — structured results
    artifacts/lit_spider_report.md     — human-readable markdown report
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.parse import urlencode

# ── Project imports ──────────────────────────────────────────────────────────

from constants import OLLAMA_URL, CONFIRMATION_MODEL, REASONING_MODELS
from llm_common import strip_think_tags, strip_markdown_fences, parse_json_response

# ── Config ───────────────────────────────────────────────────────────────────

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

EXTRACTION_MODEL = CONFIRMATION_MODEL  # llama3.1:latest — fast, good at structured output
RATE_LIMIT_SECONDS = 0.4  # stay under 3 req/sec PubMed limit
CACHE_FILE = Path("artifacts/.lit_spider_cache.json")
CACHE_MAX_AGE_DAYS = 7

ARTIFACTS_DIR = Path("artifacts")

# ── Parameter Registry ───────────────────────────────────────────────────────

PARAMETER_QUERIES = {
    # ── High priority (no citation, ~12 params) ──────────────────────────────
    "base_replication_rate": {
        "current_value": 0.1,
        "unit": "per year",
        "location": "simulator.py:297 (embedded in derivatives())",
        "current_citation": None,
        "pubmed_query": (
            "(mitochondrial DNA replication rate) OR "
            "(mtDNA copy number turnover rate) OR "
            "(mitochondrial biogenesis rate measurement)"
        ),
        "extraction_prompt": (
            "What is the measured rate of mitochondrial DNA replication or "
            "mtDNA copy number turnover in human cells? Report any "
            "quantitative rates with units (per day, per year, half-life)."
        ),
        "priority": "high",
    },
    "ros_damage_coupling": {
        "current_value": 0.15,
        "unit": "dimensionless coupling strength",
        "location": "simulator.py:310 (ROS→damage conversion)",
        "current_citation": None,
        "pubmed_query": (
            "(reactive oxygen species mitochondrial DNA damage rate) OR "
            "(ROS induced mtDNA mutation rate) OR "
            "(oxidative stress mitochondrial genome damage quantitative)"
        ),
        "extraction_prompt": (
            "What is the quantitative rate or probability of reactive oxygen "
            "species (ROS) causing mitochondrial DNA damage or mutations? "
            "Report any numerical values for ROS-induced mtDNA damage rates."
        ),
        "priority": "high",
    },
    "apoptosis_rate": {
        "current_value": 0.02,
        "unit": "per year (energy-gated)",
        "location": "simulator.py:336 (embedded in derivatives())",
        "current_citation": None,
        "pubmed_query": (
            "(mitochondrial dysfunction apoptosis rate) OR "
            "(ATP depletion cell death rate) OR "
            "(mitochondrial apoptosis quantitative measurement human)"
        ),
        "extraction_prompt": (
            "What is the measured rate of apoptosis (programmed cell death) "
            "triggered by mitochondrial dysfunction or ATP depletion in human "
            "cells? Report any numerical rates or percentages."
        ),
        "priority": "high",
    },
    "exercise_biogenesis": {
        "current_value": 0.03,
        "unit": "dimensionless coupling strength",
        "location": "simulator.py:333 (exercise → biogenesis term)",
        "current_citation": None,
        "pubmed_query": (
            "(exercise mitochondrial biogenesis rate PGC-1alpha) OR "
            "(aerobic exercise mtDNA copy number increase) OR "
            "(exercise induced mitochondrial biogenesis quantitative human)"
        ),
        "extraction_prompt": (
            "What is the measured increase in mitochondrial biogenesis or "
            "mtDNA copy number from exercise in human tissue? Report any "
            "quantitative changes (fold-change, percentage increase, rate)."
        ),
        "priority": "high",
    },
    "rapamycin_mitophagy_boost": {
        "current_value": 0.08,
        "unit": "dimensionless (added to baseline mitophagy)",
        "location": "simulator.py:366 (rapa * 0.08 term)",
        "current_citation": None,
        "pubmed_query": (
            "(rapamycin mitophagy enhancement rate) OR "
            "(mTOR inhibition mitophagy quantitative) OR "
            "(rapamycin autophagy mitochondrial clearance measurement)"
        ),
        "extraction_prompt": (
            "How much does rapamycin (or mTOR inhibition) increase mitophagy "
            "or mitochondrial autophagy? Report any quantitative fold-change "
            "or rate enhancement values."
        ),
        "priority": "high",
    },
    "nad_quality_control_boost": {
        "current_value": 0.03,
        "unit": "dimensionless (NAD-dependent mitophagy term)",
        "location": "simulator.py:366 (nad_supp * 0.03 term)",
        "current_citation": None,
        "pubmed_query": (
            "(NAD+ sirtuin mitophagy regulation) OR "
            "(NAD mitochondrial quality control SIRT1 SIRT3) OR "
            "(nicotinamide riboside mitophagy enhancement quantitative)"
        ),
        "extraction_prompt": (
            "How does NAD+ or sirtuins (SIRT1/SIRT3) regulate mitophagy or "
            "mitochondrial quality control? Report any quantitative "
            "enhancement values for NAD-dependent mitophagy."
        ),
        "priority": "high",
    },
    "senescence_ros_multiplier": {
        "current_value": 2.0,
        "unit": "multiplier",
        "location": "simulator.py:454 (1 + 2*ros term in senescence)",
        "current_citation": None,
        "pubmed_query": (
            "(reactive oxygen species cellular senescence rate) OR "
            "(ROS induced senescence quantitative) OR "
            "(oxidative stress premature senescence dose response)"
        ),
        "extraction_prompt": (
            "How much do reactive oxygen species (ROS) increase the rate of "
            "cellular senescence? Report any quantitative dose-response "
            "relationships or fold-change values."
        ),
        "priority": "high",
    },
    "atp_relaxation_time": {
        "current_value": 1.0,
        "unit": "years",
        "location": "simulator.py:396 (1.0 * (atp_target - atp) term)",
        "current_citation": None,
        "pubmed_query": (
            "(ATP production recovery time mitochondrial) OR "
            "(cellular ATP homeostasis time constant) OR "
            "(mitochondrial ATP restoration rate after stress)"
        ),
        "extraction_prompt": (
            "How quickly does ATP production recover after mitochondrial "
            "stress or damage? Report any time constants, recovery times, "
            "or half-lives for ATP level restoration."
        ),
        "priority": "high",
    },
    "ros_relaxation_time": {
        "current_value": 1.0,
        "unit": "years",
        "location": "simulator.py:417 (1.0 * (ros_eq - ros) term)",
        "current_citation": None,
        "pubmed_query": (
            "(reactive oxygen species clearance rate cell) OR "
            "(ROS scavenging half-life mitochondrial) OR "
            "(antioxidant defense ROS clearance time quantitative)"
        ),
        "extraction_prompt": (
            "What is the clearance rate or half-life of reactive oxygen "
            "species in cells? Report any quantitative time constants for "
            "ROS turnover or scavenging rates."
        ),
        "priority": "high",
    },
    "nad_relaxation_time": {
        "current_value": 0.3,
        "unit": "years",
        "location": "simulator.py:440 (0.3 * (nad_target - nad) term)",
        "current_citation": None,
        "pubmed_query": (
            "(NAD+ turnover rate half-life human) OR "
            "(NAD biosynthesis rate measurement) OR "
            "(NAD+ pool size turnover quantitative)"
        ),
        "extraction_prompt": (
            "What is the turnover rate or half-life of NAD+ in human cells "
            "or tissues? Report any quantitative rates for NAD+ synthesis, "
            "degradation, or pool turnover."
        ),
        "priority": "high",
    },
    "yamanaka_repair_rate": {
        "current_value": 0.06,
        "unit": "dimensionless (yama * 0.06 * n_d * energy_available)",
        "location": "simulator.py:329 (Yamanaka repair term)",
        "current_citation": None,
        "pubmed_query": (
            "(Yamanaka reprogramming mtDNA repair rate) OR "
            "(partial reprogramming mitochondrial rejuvenation) OR "
            "(OSKM factors mitochondrial DNA quality improvement quantitative)"
        ),
        "extraction_prompt": (
            "What is the measured rate of mitochondrial DNA repair or "
            "rejuvenation from partial Yamanaka (OSKM) reprogramming? "
            "Report any quantitative measures of mtDNA quality improvement."
        ),
        "priority": "high",
    },
    "senolytic_clearance_rate": {
        "current_value": 0.2,
        "unit": "per year (dose-dependent)",
        "location": "simulator.py:457 (seno * 0.2 * sen term)",
        "current_citation": None,
        "pubmed_query": (
            "(senolytic drug senescent cell clearance rate) OR "
            "(dasatinib quercetin senescent cell reduction quantitative) OR "
            "(navitoclax senescent cell elimination rate measurement)"
        ),
        "extraction_prompt": (
            "What is the measured rate of senescent cell clearance by "
            "senolytic drugs (dasatinib+quercetin, navitoclax, fisetin)? "
            "Report any quantitative clearance percentages or rates."
        ),
        "priority": "high",
    },

    # ── Medium priority (has citation, refinable, ~8 params) ─────────────────
    "heteroplasmy_cliff": {
        "current_value": 0.70,
        "unit": "fraction",
        "location": "constants.py:HETEROPLASMY_CLIFF",
        "current_citation": "Rossignol et al. 2003",
        "pubmed_query": (
            "(mitochondrial heteroplasmy threshold) OR "
            "(biochemical threshold heteroplasmy) OR "
            "(mitochondrial DNA mutation threshold cellular dysfunction)"
        ),
        "extraction_prompt": (
            "What fraction (percentage) of mutant/damaged mitochondrial DNA "
            "causes cellular dysfunction or ATP production failure? "
            "Report any numerical threshold values mentioned."
        ),
        "priority": "medium",
    },
    "damaged_replication_advantage": {
        "current_value": 1.05,
        "unit": "multiplier (vs healthy mtDNA replication rate)",
        "location": "constants.py:DAMAGED_REPLICATION_ADVANTAGE",
        "current_citation": "Vandiver et al. 2023 (Cramer Appendix 2 pp.154-155)",
        "pubmed_query": (
            "(mitochondrial DNA deletion replication advantage) OR "
            "(deleted mtDNA replication rate faster) OR "
            "(mitochondrial DNA deletion selfish replication quantitative)"
        ),
        "extraction_prompt": (
            "How much faster do deleted or shortened mitochondrial DNA "
            "molecules replicate compared to full-length mtDNA? Report "
            "any quantitative replication advantage (fold-change, percentage)."
        ),
        "priority": "medium",
    },
    "cliff_steepness": {
        "current_value": 15.0,
        "unit": "dimensionless (sigmoid steepness parameter)",
        "location": "constants.py:CLIFF_STEEPNESS",
        "current_citation": "Simulation calibration",
        "pubmed_query": (
            "(heteroplasmy threshold sigmoid shape) OR "
            "(mitochondrial biochemical threshold nonlinear) OR "
            "(heteroplasmy ATP production dose response curve)"
        ),
        "extraction_prompt": (
            "How sharp or gradual is the transition from normal to impaired "
            "mitochondrial function as heteroplasmy increases? Report any "
            "dose-response curve parameters or sigmoid shape descriptions."
        ),
        "priority": "medium",
    },
    "tissue_ros_sensitivity_brain": {
        "current_value": 1.5,
        "unit": "multiplier (vs default tissue)",
        "location": "constants.py:TISSUE_PROFILES['brain']['ros_sensitivity']",
        "current_citation": "Cramer Ch. V.J p.65 (qualitative)",
        "pubmed_query": (
            "(brain neuron oxidative stress vulnerability) OR "
            "(neuronal ROS sensitivity compared other tissues) OR "
            "(brain mitochondrial oxidative damage tissue specificity)"
        ),
        "extraction_prompt": (
            "How much more sensitive are brain/neuronal cells to reactive "
            "oxygen species (ROS) compared to other tissues? Report any "
            "quantitative comparisons (fold-difference, relative rates)."
        ),
        "priority": "medium",
    },
    "tissue_ros_sensitivity_cardiac": {
        "current_value": 1.2,
        "unit": "multiplier (vs default tissue)",
        "location": "constants.py:TISSUE_PROFILES['cardiac']['ros_sensitivity']",
        "current_citation": "Cramer Ch. VII (qualitative)",
        "pubmed_query": (
            "(cardiac mitochondrial oxidative stress vulnerability) OR "
            "(heart ROS sensitivity measurement) OR "
            "(cardiomyocyte oxidative damage susceptibility quantitative)"
        ),
        "extraction_prompt": (
            "How vulnerable is cardiac tissue to mitochondrial oxidative "
            "stress compared to other tissues? Report any quantitative "
            "comparisons of ROS sensitivity."
        ),
        "priority": "medium",
    },
    "tissue_biogenesis_brain": {
        "current_value": 0.3,
        "unit": "multiplier (vs default tissue)",
        "location": "constants.py:TISSUE_PROFILES['brain']['biogenesis_rate']",
        "current_citation": "Post-mitotic neuron biology (qualitative)",
        "pubmed_query": (
            "(neuronal mitochondrial biogenesis rate PGC-1alpha) OR "
            "(brain mitochondrial turnover rate measurement) OR "
            "(post-mitotic neuron mitochondrial biogenesis capacity)"
        ),
        "extraction_prompt": (
            "What is the rate of mitochondrial biogenesis in post-mitotic "
            "neurons compared to other tissues? Report any quantitative "
            "rates or relative comparisons."
        ),
        "priority": "medium",
    },
    "tissue_biogenesis_muscle": {
        "current_value": 1.5,
        "unit": "multiplier (vs default tissue)",
        "location": "constants.py:TISSUE_PROFILES['muscle']['biogenesis_rate']",
        "current_citation": "PGC-1alpha biology (qualitative)",
        "pubmed_query": (
            "(skeletal muscle mitochondrial biogenesis rate PGC-1alpha) OR "
            "(muscle exercise mitochondrial content increase) OR "
            "(skeletal muscle mitochondrial turnover rate quantitative)"
        ),
        "extraction_prompt": (
            "What is the rate of mitochondrial biogenesis in skeletal muscle, "
            "especially in response to exercise (PGC-1alpha pathway)? Report "
            "any quantitative rates or fold-changes compared to baseline."
        ),
        "priority": "medium",
    },
    "cd38_base_survival": {
        "current_value": 0.4,
        "unit": "fraction surviving CD38 degradation",
        "location": "constants.py:CD38_BASE_SURVIVAL",
        "current_citation": "Cramer Ch. VI.A.3 p.73 (qualitative)",
        "pubmed_query": (
            "(CD38 NAD+ degradation rate) OR "
            "(CD38 NMN NR degradation quantitative) OR "
            "(CD38 enzyme NAD precursor destruction measurement)"
        ),
        "extraction_prompt": (
            "What fraction of NMN or NR supplements is degraded by CD38 "
            "before reaching cells? Report any quantitative measurements "
            "of CD38-mediated NAD+ precursor degradation rates."
        ),
        "priority": "medium",
    },

    # ── Low priority (well-grounded, verify, ~6 params) ─────────────────────
    "doubling_time_young": {
        "current_value": 11.8,
        "unit": "years",
        "location": "constants.py:DOUBLING_TIME_YOUNG",
        "current_citation": "Vandiver et al. 2023 (Cramer Appendix 2 p.155)",
        "pubmed_query": (
            "(mitochondrial DNA deletion accumulation rate age) OR "
            "(mtDNA deletion doubling time human aging) OR "
            "(Vandiver 2023 mitochondrial deletion aging cell)"
        ),
        "extraction_prompt": (
            "What is the doubling time for mitochondrial DNA deletions in "
            "humans before age 65? Report any quantitative rates of mtDNA "
            "deletion accumulation with age."
        ),
        "priority": "low",
    },
    "doubling_time_old": {
        "current_value": 3.06,
        "unit": "years",
        "location": "constants.py:DOUBLING_TIME_OLD",
        "current_citation": "Vandiver et al. 2023 (Cramer Appendix 2 p.155)",
        "pubmed_query": (
            "(mtDNA deletion accumulation elderly accelerated) OR "
            "(mitochondrial DNA damage rate old age) OR "
            "(Vandiver 2023 aging cell mitochondrial deletion doubling)"
        ),
        "extraction_prompt": (
            "What is the doubling time for mitochondrial DNA deletions in "
            "humans after age 65? Report any quantitative rates of mtDNA "
            "deletion accumulation in elderly individuals."
        ),
        "priority": "low",
    },
    "nad_decline_rate": {
        "current_value": 0.01,
        "unit": "per year",
        "location": "constants.py:NAD_DECLINE_RATE",
        "current_citation": "Camacho-Pereira et al. 2016 (Cramer Ch. VI.A.3)",
        "pubmed_query": (
            "(NAD+ age-dependent decline rate human) OR "
            "(NAD+ level aging quantitative measurement) OR "
            "(Camacho-Pereira 2016 NAD decline aging)"
        ),
        "extraction_prompt": (
            "What is the measured rate of NAD+ decline with age in humans? "
            "Report any quantitative rates (per year, per decade, percentage "
            "decline) and the tissues measured."
        ),
        "priority": "low",
    },
    "senescence_rate": {
        "current_value": 0.005,
        "unit": "per year",
        "location": "constants.py:SENESCENCE_RATE",
        "current_citation": "Cramer Ch. VII.A pp.89-92 (qualitative)",
        "pubmed_query": (
            "(cellular senescence accumulation rate aging human) OR "
            "(senescent cell fraction increase age quantitative) OR "
            "(p16 senescence marker accumulation rate tissue)"
        ),
        "extraction_prompt": (
            "What is the rate of senescent cell accumulation with age in "
            "human tissues? Report any quantitative rates (per year, per "
            "decade) for p16+ or SA-beta-gal+ cell fraction increase."
        ),
        "priority": "low",
    },
    "baseline_ros": {
        "current_value": 0.1,
        "unit": "normalized",
        "location": "constants.py:BASELINE_ROS",
        "current_citation": "Cramer Ch. IV.B p.53, Ch. II.H p.14",
        "pubmed_query": (
            "(mitochondrial ROS production rate baseline) OR "
            "(electron transport chain superoxide production rate) OR "
            "(mitochondrial hydrogen peroxide production quantitative)"
        ),
        "extraction_prompt": (
            "What is the baseline rate of mitochondrial ROS (superoxide, "
            "hydrogen peroxide) production in healthy human cells? Report "
            "any quantitative production rates with units."
        ),
        "priority": "low",
    },
    "ros_per_damaged": {
        "current_value": 0.3,
        "unit": "dimensionless coupling strength",
        "location": "constants.py:ROS_PER_DAMAGED",
        "current_citation": "Cramer Ch. II.H p.14, Appendix 2 pp.152-153",
        "pubmed_query": (
            "(damaged mitochondria ROS production increase) OR "
            "(mitochondrial dysfunction electron leak superoxide) OR "
            "(mtDNA mutation ROS production fold change quantitative)"
        ),
        "extraction_prompt": (
            "How much more ROS do damaged mitochondria produce compared to "
            "healthy ones? Report any quantitative fold-change or percentage "
            "increase in ROS production from mitochondrial damage."
        ),
        "priority": "low",
    },
}

# ── PubMed API ───────────────────────────────────────────────────────────────


def pubmed_search(query: str, max_results: int = 15) -> list[str]:
    """Search PubMed, return list of PMIDs."""
    params = urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{PUBMED_ESEARCH}?{params}"
    try:
        r = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  [WARN] PubMed search failed: {e}")
        return []


def pubmed_fetch(pmids: list[str]) -> str:
    """Fetch full records (XML) for given PMIDs."""
    if not pmids:
        return ""
    params = urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    })
    url = f"{PUBMED_EFETCH}?{params}"
    try:
        r = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return ""
        return r.stdout
    except Exception as e:
        print(f"  [WARN] PubMed fetch failed: {e}")
        return ""


# ── XML Parsing ──────────────────────────────────────────────────────────────


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML into list of paper dicts."""
    if not xml_text or not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        paper = {}

        # PMID
        pmid_el = article.find(".//PMID")
        paper["pmid"] = pmid_el.text if pmid_el is not None else ""

        # Title
        title_el = article.find(".//ArticleTitle")
        paper["title"] = title_el.text if title_el is not None else ""

        # Abstract
        abstract_parts = []
        for abs_text in article.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = "".join(abs_text.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        paper["abstract"] = " ".join(abstract_parts)

        # Authors (first 3)
        authors = []
        for author in article.findall(".//Author")[:3]:
            last = author.find("LastName")
            init = author.find("Initials")
            if last is not None:
                name = last.text
                if init is not None:
                    name += f" {init.text}"
                authors.append(name)
        paper["authors"] = authors

        # Year
        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")
        paper["year"] = year_el.text[:4] if year_el is not None else ""

        # Journal
        journal_el = article.find(".//Journal/Title")
        paper["journal"] = journal_el.text if journal_el is not None else ""

        # DOI
        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
                break
        paper["doi"] = doi

        papers.append(paper)

    return papers


# ── LLM Extraction ──────────────────────────────────────────────────────────


def query_ollama_extract(prompt: str, timeout: int = 120) -> str | None:
    """Query Ollama for value extraction. Returns raw response or None."""
    payload = json.dumps({
        "model": EXTRACTION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if "error" in data:
            return None
        return data.get("response", "")
    except Exception:
        return None


def check_ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "http://localhost:11434/api/tags"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "200"
    except Exception:
        return False


def extract_values(abstract: str, param_name: str,
                   extraction_prompt: str) -> dict | None:
    """Use Ollama to extract numerical values from an abstract."""
    if not abstract.strip():
        return None

    prompt = f"""You are a biomedical research assistant. Read this abstract and answer the question.

ABSTRACT:
{abstract[:3000]}

QUESTION: {extraction_prompt}

Respond with a JSON object:
{{
  "found_values": [
    {{"value": <number>, "unit": "<unit>", "context": "<brief context>", "confidence": "high|medium|low"}}
  ],
  "relevant": true,
  "notes": "<any caveats>"
}}

If the abstract does not contain relevant numerical data, set "relevant": false and "found_values": [].
Respond ONLY with the JSON object, no other text."""

    raw = query_ollama_extract(prompt)
    if raw is None:
        return None

    cleaned = strip_think_tags(raw)
    cleaned = strip_markdown_fences(cleaned)
    parsed = parse_json_response(cleaned)
    if parsed is None:
        return {"raw_response": raw[:500], "parse_error": True,
                "found_values": [], "relevant": False}
    return parsed


# ── Keyword Fallback Extraction ──────────────────────────────────────────────

# Pattern: number optionally followed by unit
_VALUE_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*'
    r'(%|fold|×|x|years?|yr|/yr|per\s+year|per\s+day|/day|'
    r'hours?|hr|minutes?|min|seconds?|sec|'
    r'mV|nmol|pmol|μmol|mmol|nM|μM|mM|'
    r'MU|copies|per\s+cell)',
    re.IGNORECASE,
)


def extract_values_keyword(abstract: str, param_name: str) -> dict:
    """Regex-based fallback extraction when Ollama is unavailable."""
    if not abstract.strip():
        return {"found_values": [], "relevant": False,
                "notes": "empty abstract", "method": "keyword"}

    matches = _VALUE_PATTERN.findall(abstract)
    if not matches:
        return {"found_values": [], "relevant": False,
                "notes": "no numeric+unit patterns found", "method": "keyword"}

    found = []
    for value_str, unit in matches[:10]:  # cap at 10
        try:
            val = float(value_str)
        except ValueError:
            continue
        found.append({
            "value": val,
            "unit": unit.strip(),
            "context": "keyword extraction (no semantic understanding)",
            "confidence": "low",
        })

    return {
        "found_values": found,
        "relevant": len(found) > 0,
        "notes": "keyword-only (no LLM)",
        "method": "keyword",
    }


# ── Caching ──────────────────────────────────────────────────────────────────


def load_cache() -> dict:
    """Load PubMed cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        # Expire old entries
        cutoff = time.time() - CACHE_MAX_AGE_DAYS * 86400
        return {k: v for k, v in data.items()
                if v.get("_cached_at", 0) > cutoff}
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    """Save PubMed cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_findings(param_key: str, spec: dict,
                       papers: list[dict],
                       extractions: list[dict]) -> dict:
    """Aggregate LLM extractions into a parameter report."""
    relevant_papers = []
    all_values = []

    for paper, extraction in zip(papers, extractions):
        if extraction and extraction.get("relevant"):
            relevant_papers.append({
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "doi": paper.get("doi", ""),
                "authors": paper.get("authors", []),
            })
            for fv in extraction.get("found_values", []):
                all_values.append({
                    **fv,
                    "pmid": paper.get("pmid", ""),
                    "paper_title": paper.get("title", ""),
                })

    # Compute literature range
    numeric_vals = [v["value"] for v in all_values
                    if isinstance(v.get("value"), (int, float))]
    lit_range = {}
    if numeric_vals:
        lit_range = {
            "min": min(numeric_vals),
            "max": max(numeric_vals),
            "median": median(numeric_vals),
            "n": len(numeric_vals),
        }

    # Assessment
    n_relevant = len(relevant_papers)
    n_values = len(numeric_vals)
    if n_values == 0:
        assessment = "no-data"
    elif n_values <= 2:
        assessment = "sparse"
    elif n_values >= 3 and lit_range:
        spread = lit_range["max"] - lit_range["min"]
        center = lit_range["median"]
        if center != 0 and spread / abs(center) > 2.0:
            assessment = "conflicting"
        else:
            assessment = "well-supported"
    else:
        assessment = "sparse"

    # Discrepancy
    discrepancy = "none"
    if lit_range and numeric_vals:
        current = spec["current_value"]
        med = lit_range["median"]
        if med != 0:
            ratio = abs(current - med) / abs(med)
            if ratio > 1.0:
                discrepancy = "major"
            elif ratio > 0.3:
                discrepancy = "minor"
        elif current != 0:
            discrepancy = "major"

    return {
        "parameter": param_key,
        "current_value": spec["current_value"],
        "current_unit": spec["unit"],
        "location": spec["location"],
        "current_citation": spec["current_citation"],
        "priority": spec["priority"],
        "n_papers_searched": len(papers),
        "n_papers_relevant": n_relevant,
        "n_values_extracted": n_values,
        "literature_values": all_values,
        "literature_range": lit_range,
        "assessment": assessment,
        "discrepancy": discrepancy,
        "key_papers": relevant_papers[:5],
    }


# ── Report Generation ────────────────────────────────────────────────────────


def write_markdown_report(results: dict, output_path: Path) -> None:
    """Write human-readable markdown report."""
    lines = []
    now = results.get("date", "")
    lines.append("# Lit Spider Report")
    lines.append("")
    lines.append(f"**Date:** {now}")
    lines.append(f"**Model:** {results.get('ollama_model', 'N/A')}")
    lines.append(f"**Parameters searched:** {results.get('n_parameters', 0)}")
    lines.append(f"**PubMed queries:** {results.get('n_pubmed_queries', 0)}")
    lines.append(f"**Abstracts fetched:** {results.get('n_abstracts_fetched', 0)}")
    lines.append(f"**LLM extractions:** {results.get('n_llm_extractions', 0)}")
    lines.append(f"**Elapsed:** {results.get('elapsed_seconds', 0):.0f}s")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Parameter | Current | Priority | Assessment | Discrepancy | Lit Range | Papers |")
    lines.append("|-----------|---------|----------|------------|-------------|-----------|--------|")

    params = results.get("parameters", {})
    # Sort by priority then discrepancy
    priority_order = {"high": 0, "medium": 1, "low": 2}
    disc_order = {"major": 0, "minor": 1, "none": 2}
    sorted_keys = sorted(
        params.keys(),
        key=lambda k: (priority_order.get(params[k]["priority"], 9),
                        disc_order.get(params[k]["discrepancy"], 9)),
    )

    for key in sorted_keys:
        p = params[key]
        lit_str = ""
        lr = p.get("literature_range", {})
        if lr:
            lit_str = f"{lr.get('min', '?'):.3g}–{lr.get('max', '?'):.3g}"
        disc_emoji = {"major": "**MAJOR**", "minor": "minor", "none": "ok"
                      }.get(p["discrepancy"], "?")
        lines.append(
            f"| `{key}` | {p['current_value']} {p['current_unit']} | "
            f"{p['priority']} | {p['assessment']} | {disc_emoji} | "
            f"{lit_str} | {p['n_papers_relevant']} |"
        )

    lines.append("")

    # Detailed sections by priority
    for priority_label, priority_val in [("High Priority (no citation)", "high"),
                                          ("Medium Priority (refinable)", "medium"),
                                          ("Low Priority (verify)", "low")]:
        priority_params = [k for k in sorted_keys if params[k]["priority"] == priority_val]
        if not priority_params:
            continue

        lines.append(f"## {priority_label}")
        lines.append("")

        for key in priority_params:
            p = params[key]
            lines.append(f"### `{key}`")
            lines.append("")
            lines.append(f"- **Current value:** {p['current_value']} {p['current_unit']}")
            lines.append(f"- **Location:** `{p['location']}`")
            lines.append(f"- **Citation:** {p['current_citation'] or 'None'}")
            lines.append(f"- **Assessment:** {p['assessment']}")
            lines.append(f"- **Discrepancy:** {p['discrepancy']}")
            lr = p.get("literature_range", {})
            if lr:
                lines.append(f"- **Literature range:** {lr.get('min', '?'):.4g} – "
                             f"{lr.get('max', '?'):.4g} (median {lr.get('median', '?'):.4g}, "
                             f"n={lr.get('n', 0)})")
            else:
                lines.append("- **Literature range:** No numerical values found")
            lines.append("")

            # Key papers
            kp = p.get("key_papers", [])
            if kp:
                lines.append("**Key papers:**")
                lines.append("")
                for paper in kp:
                    authors = ", ".join(paper.get("authors", []))
                    year = paper.get("year", "")
                    title = paper.get("title", "")
                    pmid = paper.get("pmid", "")
                    doi = paper.get("doi", "")
                    doi_link = f" [DOI](https://doi.org/{doi})" if doi else ""
                    lines.append(f"- {authors} ({year}). *{title}* "
                                 f"[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                                 f"{doi_link}")
                lines.append("")

            # Extracted values
            lit_vals = p.get("literature_values", [])
            if lit_vals:
                lines.append("**Extracted values:**")
                lines.append("")
                for v in lit_vals[:8]:
                    conf = v.get("confidence", "?")
                    ctx = v.get("context", "")
                    lines.append(f"- {v.get('value', '?')} {v.get('unit', '')} "
                                 f"({conf}) — {ctx}")
                lines.append("")

    # Summary categories
    summary = results.get("summary", {})
    lines.append("## Assessment Categories")
    lines.append("")
    for cat, label in [("major_discrepancies", "Major Discrepancies (action needed)"),
                        ("well_supported", "Well Supported"),
                        ("sparse_evidence", "Sparse Evidence"),
                        ("no_data", "No Data Found")]:
        items = summary.get(cat, [])
        lines.append(f"### {label}")
        if items:
            for item in items:
                lines.append(f"- `{item}`")
        else:
            lines.append("- (none)")
        lines.append("")

    output_path.write_text("\n".join(lines))


# ── Main Pipeline ────────────────────────────────────────────────────────────


def run_lit_spider(
    params: list[str] | None = None,
    max_papers: int = 15,
    use_llm: bool = True,
    use_cache: bool = True,
) -> dict:
    """Run the Lit Spider pipeline.

    Args:
        params: Subset of PARAMETER_QUERIES keys, or None for all.
        max_papers: Max PubMed papers per parameter.
        use_llm: Whether to use Ollama for extraction (falls back to keyword).
        use_cache: Whether to use PubMed result cache.

    Returns:
        Results dict (also written to artifacts/).
    """
    t0 = time.time()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Select parameters
    if params:
        query_keys = [k for k in params if k in PARAMETER_QUERIES]
        if not query_keys:
            print(f"[ERROR] No valid parameter keys in: {params}")
            print(f"  Available: {list(PARAMETER_QUERIES.keys())}")
            sys.exit(1)
    else:
        query_keys = list(PARAMETER_QUERIES.keys())

    print(f"Lit Spider — {len(query_keys)} parameters to search")
    print(f"  Max papers per param: {max_papers}")

    # Check Ollama
    ollama_ok = False
    if use_llm:
        ollama_ok = check_ollama_available()
        if ollama_ok:
            print(f"  LLM extraction: {EXTRACTION_MODEL}")
        else:
            print("  [WARN] Ollama not available — falling back to keyword extraction")
    else:
        print("  LLM extraction: disabled (keyword-only mode)")

    # Load cache
    cache = load_cache() if use_cache else {}

    # Results accumulator
    param_results = {}
    total_abstracts = 0
    total_extractions = 0

    for i, key in enumerate(query_keys):
        spec = PARAMETER_QUERIES[key]
        print(f"\n[{i+1}/{len(query_keys)}] {key} (priority: {spec['priority']})")

        # PubMed search
        cache_key = f"{spec['pubmed_query']}|{max_papers}"
        if use_cache and cache_key in cache:
            pmids = cache[cache_key]["pmids"]
            print(f"  PubMed: {len(pmids)} PMIDs (cached)")
        else:
            pmids = pubmed_search(spec["pubmed_query"], max_results=max_papers)
            print(f"  PubMed: {len(pmids)} PMIDs")
            if use_cache:
                cache[cache_key] = {
                    "pmids": pmids,
                    "_cached_at": time.time(),
                }
            time.sleep(RATE_LIMIT_SECONDS)

        if not pmids:
            param_results[key] = aggregate_findings(key, spec, [], [])
            continue

        # Fetch abstracts
        xml_text = pubmed_fetch(pmids)
        time.sleep(RATE_LIMIT_SECONDS)
        papers = parse_pubmed_xml(xml_text)
        print(f"  Fetched: {len(papers)} papers")
        total_abstracts += len(papers)

        # Extract values from each paper
        extractions = []
        for j, paper in enumerate(papers):
            abstract = paper.get("abstract", "")
            if not abstract.strip():
                extractions.append(None)
                continue

            if ollama_ok and use_llm:
                ext = extract_values(abstract, key, spec["extraction_prompt"])
                if ext and ext.get("relevant"):
                    n_found = len(ext.get("found_values", []))
                    print(f"    Paper {j+1}: {n_found} values extracted")
                total_extractions += 1
            else:
                ext = extract_values_keyword(abstract, key)

            extractions.append(ext)

        # Aggregate
        findings = aggregate_findings(key, spec, papers, extractions)
        param_results[key] = findings
        print(f"  Result: {findings['assessment']} "
              f"({findings['n_values_extracted']} values, "
              f"discrepancy: {findings['discrepancy']})")

    # Save cache
    if use_cache:
        save_cache(cache)

    # Build summary
    summary = {
        "well_supported": [],
        "sparse_evidence": [],
        "no_data": [],
        "major_discrepancies": [],
        "conflicting": [],
    }
    for key, findings in param_results.items():
        assessment = findings["assessment"]
        if assessment == "well-supported":
            summary["well_supported"].append(key)
        elif assessment == "sparse":
            summary["sparse_evidence"].append(key)
        elif assessment == "no-data":
            summary["no_data"].append(key)
        elif assessment == "conflicting":
            summary["conflicting"].append(key)
        if findings["discrepancy"] == "major":
            summary["major_discrepancies"].append(key)

    elapsed = time.time() - t0

    results = {
        "experiment": "lit_spider",
        "date": now,
        "elapsed_seconds": round(elapsed, 1),
        "n_parameters": len(query_keys),
        "n_pubmed_queries": len(query_keys),
        "n_abstracts_fetched": total_abstracts,
        "n_llm_extractions": total_extractions,
        "ollama_model": EXTRACTION_MODEL if (ollama_ok and use_llm) else "keyword-only",
        "parameters": param_results,
        "summary": summary,
    }

    # Write outputs
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ARTIFACTS_DIR / "lit_spider.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nJSON: {json_path}")

    md_path = ARTIFACTS_DIR / "lit_spider_report.md"
    write_markdown_report(results, md_path)
    print(f"Report: {md_path}")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Well-supported: {len(summary['well_supported'])}")
    print(f"  Sparse: {len(summary['sparse_evidence'])}")
    print(f"  No data: {len(summary['no_data'])}")
    print(f"  Major discrepancies: {len(summary['major_discrepancies'])}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lit Spider — PubMed literature search for simulator parameters",
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="Comma-separated parameter keys (default: all)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=15,
        help="Max PubMed papers per parameter (default: 15)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM extraction (keyword-only mode)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable PubMed cache",
    )
    args = parser.parse_args()

    param_list = None
    if args.params:
        param_list = [p.strip() for p in args.params.split(",")]

    run_lit_spider(
        params=param_list,
        max_papers=args.max_papers,
        use_llm=not args.no_llm,
        use_cache=not args.no_cache,
    )
