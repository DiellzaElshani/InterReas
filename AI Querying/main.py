import json
import re
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from rdflib import Graph, RDF, OWL, URIRef
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from prompt_templates import (
    build_count_system,
    build_ask_system,
    build_select_list_system,
    build_ordinal_select_system,
    build_select_value_system,
    build_discovery_system,
    build_repair_system,
    build_hybrid_plan_system,
    build_graph_simplifier_system,
)

# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-5.4"

REPO_ID = "12"
ENDPOINT = f"http://localhost:7200/repositories/{REPO_ID}"

ONTOLOGY_TTL_PATH = Path(r"C:\Users\User\Desktop\All\ITECH\Hiwi\Ontology\TBO Ontology_V5.ttl")
VECTOR_DB_DIR = r"C:\Users\User\Desktop\All\ITECH\Hiwi\AI Querying\Query Ecosystem\timber_vector_db"
EMBEDDING_MODEL = "nomic-embed-text"

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# OPENAI HELPERS
# ============================================================

def llm_text(user_prompt: str, system_prompt: str, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:json|sparql)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def llm_json(user_prompt: str, system_prompt: str) -> dict[str, Any]:
    raw = llm_text(user_prompt, system_prompt, temperature=0.0)
    raw = strip_markdown_fences(raw)

    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from model output:\n{raw}")

# ============================================================
# ROUTING
# ============================================================

def get_routing_decision(user_query: str) -> str:
    q = user_query.lower()

    vector_triggers = [
        "brochure", "manual", "regulation", "standard", "standards",
        "requirement", "requirements", "guideline", "guidelines",
        "code", "codes", "approval", "compliance", "compliant",
        "considerations", "eurocode", "din", "german", "fire design",
        "what should i pay attention to", "what should i consider",
        "what additional", "what information would i need"
    ]

    graph_triggers = [
        "how many", "count", "number of", "list", "show me", "which",
        "what material", "fire rating", "material", "materials", "supports",
        "supported", "load bearing", "load-bearing", "beam", "column",
        "columns", "wall", "walls", "slab", "slabs", "panel", "panels",
        "element", "elements", "id", "ids", "graph data", "graph fire-rating data"
    ]

    needs_vector = any(term in q for term in vector_triggers)
    needs_graph = any(term in q for term in graph_triggers)

    if needs_graph and needs_vector:
        return "HYBRID"
    if needs_graph:
        return "SPARQL_ONLY"
    return "VECTOR_ONLY"

# ============================================================
# ONTOLOGY HELPERS
# ============================================================

def load_ontology(ttl_path: Path) -> Graph:
    g = Graph()
    g.parse(ttl_path)
    return g


def extract_ttl_prefix_lines(ttl_path: Path) -> list[str]:
    prefix_lines = []

    with open(ttl_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("@prefix"):
                prefix_lines.append(stripped)
            elif stripped and not stripped.startswith("@"):
                break

    return prefix_lines


def ttl_prefix_lines_to_sparql_prefixes(prefix_lines: list[str]) -> list[str]:
    sparql_lines = []

    for line in prefix_lines:
        match = re.match(r"@prefix\s+([A-Za-z0-9_-]+:)\s*<([^>]+)>\s*\.", line)
        if match:
            sparql_lines.append(f"PREFIX {match.group(1)} <{match.group(2)}>")

    return sparql_lines


def get_raw_ttl_prefix_text(ttl_path: Path) -> str:
    return "\n".join(extract_ttl_prefix_lines(ttl_path))


def get_sparql_prefix_text_from_ttl(ttl_path: Path) -> str:
    return "\n".join(ttl_prefix_lines_to_sparql_prefixes(extract_ttl_prefix_lines(ttl_path)))


def extract_classes(graph: Graph) -> list[str]:
    classes = set()

    for cls in graph.subjects(RDF.type, OWL.Class):
        try:
            classes.add(graph.namespace_manager.normalizeUri(cls))
        except Exception:
            continue

    return sorted(classes)


def extract_predicates(graph: Graph) -> list[str]:
    predicates = set()

    for _, predicate, _ in graph:
        try:
            predicates.add(graph.namespace_manager.normalizeUri(predicate))
        except Exception:
            continue

    return sorted(predicates)


def extract_named_individuals(graph: Graph) -> list[str]:
    individuals = set()

    for s, _, _ in graph:
        if isinstance(s, URIRef):
            try:
                individuals.add(graph.namespace_manager.normalizeUri(s))
            except Exception:
                continue

    return sorted(individuals)

# ============================================================
# TEXT / VOCAB HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def local_name(term: str) -> str:
    return term.split(":", 1)[1] if ":" in term else term


def humanize_term(term: str) -> str:
    name = local_name(term)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ").replace("-", " ")
    return normalize_text(name)


def similarity_score(phrase: str, term: str) -> int:
    phrase_tokens = set(tokenize(phrase))
    term_tokens = set(tokenize(humanize_term(term)))
    if not phrase_tokens or not term_tokens:
        return 0

    score = 0
    overlap = phrase_tokens & term_tokens
    score += 10 * len(overlap)

    phrase_norm = re.sub(r"[_\-\s]", "", phrase.lower())
    term_norm = re.sub(r"[_\-\s]", "", humanize_term(term).lower())

    if phrase_norm == term_norm:
        score += 25
    elif phrase_norm in term_norm or term_norm in phrase_norm:
        score += 8

    return score


def top_matches(phrase: str, terms: list[str], limit: int = 5, min_score: int = 1) -> list[str]:
    scored = []
    for term in terms:
        score = similarity_score(phrase, term)
        if score >= min_score:
            scored.append((score, term))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [term for _, term in scored[:limit]]

# ============================================================
# BASIC QUESTION ANALYSIS
# ============================================================

def classify_question_rule_based(user_query: str) -> str:
    q = normalize_text(user_query)

    if any(term in q for term in ["how many", "count", "number of"]):
        return "count"

    if q.startswith(("is there", "are there", "does any", "do any", "has any", "have any")):
        return "ask"

    # explicit multi-instance retrieval
    if any(term in q for term in ["list", "show me", "which"]):
        return "select_list"

    if "what values" in q or "what data" in q or "what properties" in q:
        return "select_list"

    # single value style questions
    if q.startswith("what is ") or q.startswith("what are "):
        return "select_value"

    if q.startswith("what "):
        return "select_value"

    return "discovery"


def has_ordinal_language(user_query: str) -> bool:
    q = user_query.lower()
    ordinal_terms = ["first", "second", "third", "last", "earliest", "latest", "top", "bottom"]
    return any(term in q for term in ordinal_terms)


def tokenize_query_for_ids(query: str) -> list[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", query)


def find_grounded_entities(user_query: str, known_individuals: list[str]) -> dict[str, str]:
    grounded = {}
    query_tokens = tokenize_query_for_ids(user_query)

    individual_lookup = []
    for term in known_individuals:
        individual_lookup.append((term, humanize_term(term)))

    for token in query_tokens:
        token_norm = re.sub(r"[_\-\s]", "", token.lower())

        exact_matches = []
        for full, local_human in individual_lookup:
            local_norm = re.sub(r"[_\-\s]", "", local_human)
            if local_norm == token_norm:
                exact_matches.append(full)

        if len(exact_matches) == 1:
            grounded[token] = exact_matches[0]

    return grounded


def build_grounded_entities_text(grounded_entities: dict[str, str]) -> str:
    if not grounded_entities:
        return "None"

    lines = []
    for raw, grounded in grounded_entities.items():
        lines.append(f"- {raw} -> {grounded}")
    return "\n".join(lines)

# ============================================================
# HYBRID PLANNING
# ============================================================

def default_hybrid_plan(user_query: str) -> dict[str, Any]:
    q = user_query.lower()
    task_type = "compliance_check" if any(x in q for x in ["compliant", "compliance", "standard", "standards", "code", "regulation"]) else "explain_from_graph_fact"

    return {
        "task_type": task_type,
        "graph_question": "",
        "document_question": user_query,
        "jurisdiction": "Germany" if "german" in q or "germany" in q else "",
    }


def normalize_hybrid_plan(plan: dict[str, Any], user_query: str) -> dict[str, Any]:
    fixed = default_hybrid_plan(user_query)

    if isinstance(plan, dict):
        if isinstance(plan.get("task_type"), str) and plan["task_type"].strip():
            fixed["task_type"] = plan["task_type"].strip()

        if isinstance(plan.get("graph_question"), str) and plan["graph_question"].strip():
            fixed["graph_question"] = plan["graph_question"].strip()

        if isinstance(plan.get("document_question"), str) and plan["document_question"].strip():
            fixed["document_question"] = plan["document_question"].strip()

        if isinstance(plan.get("jurisdiction"), str) and plan["jurisdiction"].strip():
            fixed["jurisdiction"] = plan["jurisdiction"].strip()

    allowed_tasks = {"explain_from_graph_fact", "compliance_check", "insufficient_data_check"}
    if fixed["task_type"] not in allowed_tasks:
        fixed["task_type"] = default_hybrid_plan(user_query)["task_type"]

    if not fixed["graph_question"]:
        fixed["graph_question"] = user_query

    if not fixed["document_question"]:
        fixed["document_question"] = user_query

    return fixed


def plan_hybrid_query(user_query: str) -> dict[str, Any]:
    system_prompt = build_hybrid_plan_system()
    plan = llm_json(user_query, system_prompt)
    return normalize_hybrid_plan(plan, user_query)


def simplify_graph_question(user_query: str, hybrid_plan: dict[str, Any]) -> str:
    system_prompt = build_graph_simplifier_system()
    prompt = f"""
Original user question:
{user_query}

Hybrid plan:
{json.dumps(hybrid_plan, indent=2)}
""".strip()

    simplified = llm_text(prompt, system_prompt, temperature=0.0)
    simplified = strip_markdown_fences(simplified).strip()
    if not simplified:
        return hybrid_plan.get("graph_question", user_query)

    original_graph_question = hybrid_plan.get("graph_question", user_query).strip()

    # preserve original query type
    original_type = classify_question_rule_based(original_graph_question)
    simplified_type = classify_question_rule_based(simplified)

    original_ordinal = has_ordinal_language(original_graph_question)
    simplified_ordinal = has_ordinal_language(simplified)

    if original_type != simplified_type or original_ordinal != simplified_ordinal:
        return original_graph_question

    return simplified

# ============================================================
# PROMPT ROUTING
# ============================================================

def get_generation_system(
    q_type: str,
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
    ordinal_query: bool = False,
) -> str:
    if ordinal_query:
        return build_ordinal_select_system(
            prefix_block,
            classes,
            predicates,
            ttl_prefix_text,
            grounded_entities_text,
        )

    if q_type == "count":
        return build_count_system(prefix_block, classes, predicates, ttl_prefix_text, grounded_entities_text)
    if q_type == "ask":
        return build_ask_system(prefix_block, classes, predicates, ttl_prefix_text, grounded_entities_text)
    if q_type == "select_list":
        return build_select_list_system(prefix_block, classes, predicates, ttl_prefix_text, grounded_entities_text)
    if q_type == "select_value":
        return build_select_value_system(prefix_block, classes, predicates, ttl_prefix_text, grounded_entities_text)
    return build_discovery_system(prefix_block, ttl_prefix_text, grounded_entities_text)

# ============================================================
# DOCUMENT QUERY BUILDING
# ============================================================

def build_document_query_from_plan(plan: dict[str, Any], simplified_graph_question: str, graph_context: str) -> str:
    task = plan.get("task_type", "explain_from_graph_fact")
    jurisdiction = plan.get("jurisdiction", "")
    document_question = plan.get("document_question", "")

    if task == "compliance_check":
        return f"""
Task: compliance check
Jurisdiction: {jurisdiction}

Original document-side intent:
{document_question}

Graph-side question:
{simplified_graph_question}

Graph facts found:
{graph_context}

Retrieve short passages that give:
- governing standard/framework
- comparison criteria or requirements
- any limitation on making a compliance judgment
""".strip()

    return f"""
Task: explain design considerations
Jurisdiction: {jurisdiction}

Original document-side intent:
{document_question}

Graph-side question:
{simplified_graph_question}

Graph facts found:
{graph_context}

Retrieve short passages that explain what should be checked or considered based on those graph facts.
""".strip()

# ============================================================
# SPARQL GENERATION / CLEANUP
# ============================================================

def generate_sparql(user_query: str, system_prompt: str) -> str:
    return llm_text(user_query, system_prompt, temperature=0.0)


def repair_sparql(
    user_query: str,
    broken_query: str,
    error_text: str,
    sparql_prefixes: str,
    classes: list[str],
    predicates: list[str],
    grounded_entities_text: str,
) -> str:
    system_prompt = build_repair_system(
        prefix_block=sparql_prefixes,
        classes=classes,
        predicates=predicates,
        grounded_entities_text=grounded_entities_text,
    )

    repair_prompt = f"""
Original user question:
{user_query}

Broken SPARQL:
{broken_query}

Error or issue:
{error_text}
""".strip()

    return llm_text(repair_prompt, system_prompt, temperature=0.0)


def extract_first_query_block(text: str) -> str:
    text = strip_markdown_fences(text)

    prefix_matches = list(re.finditer(r"PREFIX\s+[A-Za-z0-9_-]+:\s*<[^>]+>", text, flags=re.IGNORECASE))
    select_match = re.search(r"\bSELECT\b|\bASK\b", text, flags=re.IGNORECASE)

    if prefix_matches and select_match:
        start = prefix_matches[0].start()
        return text[start:].strip()

    if select_match:
        return text[select_match.start():].strip()

    return text.strip()


def remove_duplicate_prefixes(query: str) -> str:
    lines = query.splitlines()
    prefix_seen = set()
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("PREFIX "):
            if stripped not in prefix_seen:
                cleaned.append(stripped)
                prefix_seen.add(stripped)
        else:
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def get_used_prefixes_from_query(query: str) -> set[str]:
    used = set()

    query_without_prefix_lines = "\n".join(
        line for line in query.splitlines()
        if not line.strip().upper().startswith("PREFIX ")
    )

    tokens = re.findall(r"\b([A-Za-z0-9_-]+):[A-Za-z0-9_-]+\b", query_without_prefix_lines)

    for token in tokens:
        if token.lower() not in {"http", "https"}:
            used.add(token + ":")

    if "rdf:type" in query_without_prefix_lines:
        used.add("rdf:")
    if "rdfs:" in query_without_prefix_lines:
        used.add("rdfs:")
    if "xsd:" in query_without_prefix_lines:
        used.add("xsd:")

    return used


def ensure_only_used_prefixes_present(query: str, ttl_path: Path) -> str:
    prefix_lines = extract_ttl_prefix_lines(ttl_path)
    ontology_prefixes = {}

    for line in prefix_lines:
        match = re.match(r"@prefix\s+([A-Za-z0-9_-]+:)\s*<([^>]+)>\s*\.", line)
        if match:
            ontology_prefixes[match.group(1)] = f"PREFIX {match.group(1)} <{match.group(2)}>"

    used_prefixes = get_used_prefixes_from_query(query)
    final_prefixes = [ontology_prefixes[p] for p in ontology_prefixes if p in used_prefixes]

    query_wo_prefixes = "\n".join(
        line for line in query.splitlines()
        if not line.strip().upper().startswith("PREFIX ")
    ).strip()

    if final_prefixes:
        return "\n".join(final_prefixes) + "\n\n" + query_wo_prefixes

    return query_wo_prefixes


def fix_invalid_prefixed_variables(query: str) -> str:
    pattern = r"\?([A-Za-z0-9_-]+:)+([A-Za-z0-9_-]+)"

    def repl(match):
        local = match.group(2)
        return "?" + (local[:1].lower() + local[1:] if local else "element")

    return re.sub(pattern, repl, query)


def collapse_repeated_prefixes(query: str) -> str:
    pattern = r"\b([A-Za-z0-9_-]+:)+([A-Za-z0-9_-]+)\b"

    def repl(match):
        full = match.group(0)
        parts = full.split(":")
        local = parts[-1]
        prefixes = parts[:-1]

        if prefixes and len(set(prefixes)) == 1:
            return f"{prefixes[0]}:{local}"

        return full

    return re.sub(pattern, repl, query)


def enforce_grounded_entities(query: str, grounded_entities: dict[str, str]) -> str:
    fixed = query

    for raw, grounded in grounded_entities.items():
        grounded_local = local_name(grounded)

        fixed = re.sub(
            rf"\b(?:[A-Za-z0-9_-]+:)+{re.escape(grounded_local)}\b",
            grounded,
            fixed,
            flags=re.IGNORECASE
        )

        fixed = re.sub(
            rf"\b{re.escape(raw)}\b",
            grounded,
            fixed,
            flags=re.IGNORECASE
        )

        fixed = re.sub(
            rf"\b{re.escape(grounded_local)}\b",
            grounded,
            fixed,
            flags=re.IGNORECASE
        )

    return fixed


def reject_id_like_variables(query: str) -> str | None:
    if re.search(r"\?[A-Za-z]+[_-][A-Za-z0-9]+\b", query):
        return "Invalid query: identifier-like token was turned into a variable instead of a grounded entity."
    return None


def validate_query_syntax_guardrails(query: str) -> str | None:
    if re.search(r"\?(?:[A-Za-z0-9_-]+:)+[A-Za-z0-9_-]+", query):
        return "Invalid SPARQL variable: variables cannot contain namespace prefixes."

    if re.search(r"\b([A-Za-z0-9_-]+:){2,}[A-Za-z0-9_-]+\b", query):
        return "Invalid prefixed name: repeated namespace prefix detected."

    id_var_error = reject_id_like_variables(query)
    if id_var_error:
        return id_var_error

    return None


def cleanup_query(
    raw_query: str,
    ttl_path: Path,
    grounded_entities: dict[str, str],
) -> str:
    query = extract_first_query_block(raw_query)
    query = fix_invalid_prefixed_variables(query)
    query = collapse_repeated_prefixes(query)
    query = enforce_grounded_entities(query, grounded_entities)
    query = fix_invalid_prefixed_variables(query)
    query = collapse_repeated_prefixes(query)
    query = remove_duplicate_prefixes(query)
    query = ensure_only_used_prefixes_present(query, ttl_path)
    return query.strip()

# ============================================================
# GRAPHDB
# ============================================================

def run_query(sparql_query: str) -> requests.Response:
    headers = {"Accept": "application/sparql-results+json"}
    return requests.post(
        ENDPOINT,
        data={"query": sparql_query},
        headers=headers,
        timeout=60,
    )


def parse_graphdb_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def simplify_binding_value(binding_value: dict) -> str:
    if not isinstance(binding_value, dict):
        return str(binding_value)
    return str(binding_value.get("value", ""))


def summarize_sparql_json(data: dict[str, Any]) -> str:
    if "boolean" in data:
        return f"ASK result: {data['boolean']}"

    results = data.get("results", {}).get("bindings", [])
    if not results:
        return "No results found."

    grouped = {}
    has_element = any("element" in row for row in results)

    if has_element:
        for row in results:
            element_value = simplify_binding_value(row.get("element", {}))
            if not element_value:
                continue
            grouped.setdefault(element_value, {})
            for var, value_dict in row.items():
                if var == "element":
                    continue
                value = simplify_binding_value(value_dict)
                if not value:
                    continue
                grouped[element_value].setdefault(var, set()).add(value)

        lines = []
        for element, props in list(grouped.items())[:20]:
            parts = [f"{k}: {', '.join(sorted(v))}" for k, v in props.items()]
            line = f"element: {element}"
            if parts:
                line += "; " + "; ".join(parts)
            lines.append(line)
        return "\n".join(lines)

    lines = []
    seen = set()
    for row in results[:50]:
        parts = []
        for var, value_dict in row.items():
            parts.append(f"{var}: {simplify_binding_value(value_dict)}")
        line = "; ".join(parts)
        if line not in seen:
            seen.add(line)
            lines.append(line)

    return "\n".join(lines[:20])

# ============================================================
# VECTOR PATH
# ============================================================

def run_vector_path(query: str) -> str:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )

    docs = vector_db.similarity_search(query, k=4)

    if not docs:
        return "No document context found."

    chunks = []
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip().replace("\n", " ")
        source = ""
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = doc.metadata.get("source", "")
        label = f"Document chunk {i}"
        if source:
            label += f" [{source}]"
        chunks.append(f"{label}: {text[:700]}")

    return "\n".join(chunks)

# ============================================================
# DISCOVERY / FALLBACK
# ============================================================

def search_entity_candidates(token: str, ttl_path: Path) -> str:
    prefix_lines = extract_ttl_prefix_lines(ttl_path)
    sparql_prefixes = ttl_prefix_lines_to_sparql_prefixes(prefix_lines)
    prefix_block = "\n".join(sparql_prefixes)

    return f"""{prefix_block}

SELECT DISTINCT ?entity
WHERE {{
  ?entity ?p ?o .
  FILTER(regex(str(?entity), "{token}", "i"))
}}
LIMIT 20
""".strip()

# ============================================================
# EXECUTION CORE
# ============================================================

def execute_generated_query(
    generation_prompt: str,
    system_prompt: str,
    ontology_context: tuple,
    grounded_entities: dict[str, str],
) -> tuple[str, str]:
    _, classes, predicates, _, sparql_prefixes, _ = ontology_context

    sparql_raw = generate_sparql(generation_prompt, system_prompt)
    sparql_clean = cleanup_query(
        raw_query=sparql_raw,
        ttl_path=ONTOLOGY_TTL_PATH,
        grounded_entities=grounded_entities,
    )

    validation_error = validate_query_syntax_guardrails(sparql_clean)
    if validation_error:
        repaired_raw = repair_sparql(
            user_query=generation_prompt,
            broken_query=sparql_clean,
            error_text=validation_error,
            sparql_prefixes=sparql_prefixes,
            classes=classes,
            predicates=predicates,
            grounded_entities_text=build_grounded_entities_text(grounded_entities),
        )
        sparql_clean = cleanup_query(
            raw_query=repaired_raw,
            ttl_path=ONTOLOGY_TTL_PATH,
            grounded_entities=grounded_entities,
        )
        validation_error = validate_query_syntax_guardrails(sparql_clean)
        if validation_error:
            return sparql_clean, f"Validation error:\n{validation_error}"

    response = run_query(sparql_clean)
    if response.status_code == 200:
        json_data = parse_graphdb_response(response)
        return sparql_clean, summarize_sparql_json(json_data)

    graph_error = response.text[:1500]

    repaired_raw = repair_sparql(
        user_query=generation_prompt,
        broken_query=sparql_clean,
        error_text=graph_error,
        sparql_prefixes=sparql_prefixes,
        classes=classes,
        predicates=predicates,
        grounded_entities_text=build_grounded_entities_text(grounded_entities),
    )
    repaired_clean = cleanup_query(
        raw_query=repaired_raw,
        ttl_path=ONTOLOGY_TTL_PATH,
        grounded_entities=grounded_entities,
    )

    validation_error = validate_query_syntax_guardrails(repaired_clean)
    if validation_error:
        return repaired_clean, f"Validation error after repair:\n{validation_error}"

    retry_response = run_query(repaired_clean)
    if retry_response.status_code != 200:
        return repaired_clean, f"GraphDB error:\n{retry_response.text[:1500]}"

    json_data = parse_graphdb_response(retry_response)
    return repaired_clean, summarize_sparql_json(json_data)

# ============================================================
# GRAPH PATHS
# ============================================================

def filter_relevant_terms(query: str, terms: list[str], limit: int = 40) -> list[str]:
    q_words = set(re.findall(r"[a-zA-Z_]+", query.lower()))
    scored = []

    for term in terms:
        term_lower = humanize_term(term)
        score = sum(1 for word in q_words if word in term_lower)
        if score > 0:
            scored.append((score, term))

    scored.sort(key=lambda x: (-x[0], x[1]))
    if scored:
        return [term for _, term in scored[:limit]]
    return terms[:limit]


def run_graph_path(query: str, ontology_context: tuple) -> tuple[str, str]:
    _, classes, predicates, individuals, sparql_prefixes, raw_prefixes = ontology_context

    q_type = classify_question_rule_based(query)
    ordinal_query = has_ordinal_language(query)
    rel_classes = filter_relevant_terms(query, classes)
    rel_preds = filter_relevant_terms(query, predicates)

    grounded_entities = find_grounded_entities(query, individuals)
    grounded_entities_text = build_grounded_entities_text(grounded_entities)

    id_like_tokens = [t for t in tokenize_query_for_ids(query) if re.search(r"[A-Za-z]+[_-]?[0-9]+", t)]
    if id_like_tokens and not grounded_entities:
        candidate_query = search_entity_candidates(id_like_tokens[0], ONTOLOGY_TTL_PATH)
        response = run_query(candidate_query)
        if response.status_code == 200:
            json_data = parse_graphdb_response(response)
            candidate_summary = summarize_sparql_json(json_data)
            return candidate_query, f"Entity grounding failed. Candidate matches for '{id_like_tokens[0]}':\n{candidate_summary}"

    system_prompt = get_generation_system(
        q_type=q_type,
        prefix_block=sparql_prefixes,
        classes=rel_classes,
        predicates=rel_preds,
        ttl_prefix_text=raw_prefixes,
        grounded_entities_text=grounded_entities_text,
        ordinal_query=ordinal_query,
    )

    return execute_generated_query(
        generation_prompt=query,
        system_prompt=system_prompt,
        ontology_context=ontology_context,
        grounded_entities=grounded_entities,
    )

# ============================================================
# FINAL ANSWERING
# ============================================================

def build_final_answer(
    user_query: str,
    route: str,
    sparql_query: str,
    graph_context: str,
    vector_context: str,
    hybrid_plan: dict[str, Any] | None = None,
    simplified_graph_question: str = "",
) -> str:
    if graph_context.startswith("Validation error") or graph_context.startswith("GraphDB error"):
        return "ANSWER:\n- Query failed.\n\nMISSING:\n- Reliable graph result."

    if route == "HYBRID":
        system_prompt = """
You answer hybrid graph + document questions for building design and compliance.

Output exactly these 4 sections:
FACTS:
- ...
DOCS:
- ...
ANSWER:
- ...
MISSING:
- ...

Rules:
- Keep it short.
- Maximum 1 bullet per section.
- Use only the provided graph facts and document context.
- If the requested graph fact was not found, say that directly.
- If the documents give only general guidance, say that directly.
- Do not invent project facts or thresholds.
""".strip()

        prompt = f"""
User question:
{user_query}

Hybrid plan:
{json.dumps(hybrid_plan or {}, indent=2)}

Simplified graph question:
{simplified_graph_question}

Generated SPARQL:
{sparql_query if sparql_query else "None"}

Graph result summary:
{graph_context if graph_context else "None"}

Document context:
{vector_context if vector_context else "None"}
""".strip()

        return llm_text(prompt, system_prompt, temperature=0.1)

    system_prompt = """
Answer the user's question using only the provided evidence.

Output exactly:
ANSWER:
- ...

Keep it short.
""".strip()

    prompt = f"""
User question:
{user_query}

Generated SPARQL:
{sparql_query if sparql_query else "None"}

Graph result summary:
{graph_context if graph_context else "None"}

Document context:
{vector_context if vector_context else "None"}
""".strip()

    return llm_text(prompt, system_prompt, temperature=0.1)

# ============================================================
# MAIN
# ============================================================

def main():
    graph = load_ontology(ONTOLOGY_TTL_PATH)

    ontology_context = (
        graph,
        extract_classes(graph),
        extract_predicates(graph),
        extract_named_individuals(graph),
        get_sparql_prefix_text_from_ttl(ONTOLOGY_TTL_PATH),
        get_raw_ttl_prefix_text(ONTOLOGY_TTL_PATH),
    )

    user_query = input("\nQuery: ").strip()
    if not user_query:
        print("No query entered.")
        return

    route = get_routing_decision(user_query)

    sparql_query = ""
    graph_context = ""
    vector_context = ""
    hybrid_plan = None
    simplified_graph_question = ""

    if route == "HYBRID":
        hybrid_plan = plan_hybrid_query(user_query)
        simplified_graph_question = simplify_graph_question(user_query, hybrid_plan)
        sparql_query, graph_context = run_graph_path(simplified_graph_question, ontology_context)
        document_query = build_document_query_from_plan(hybrid_plan, simplified_graph_question, graph_context)
        vector_context = run_vector_path(document_query)

    elif route == "SPARQL_ONLY":
        sparql_query, graph_context = run_graph_path(user_query, ontology_context)

    else:
        vector_context = run_vector_path(user_query)

    final_answer = build_final_answer(
        user_query=user_query,
        route=route,
        sparql_query=sparql_query,
        graph_context=graph_context,
        vector_context=vector_context,
        hybrid_plan=hybrid_plan,
        simplified_graph_question=simplified_graph_question,
    )

    print(f"\n[Route: {route}]")

    if hybrid_plan:
        print("\n--- HYBRID PLAN ---")
        print(json.dumps(hybrid_plan, indent=2))

    if simplified_graph_question:
        print("\n--- SIMPLIFIED GRAPH QUESTION ---")
        print(simplified_graph_question)

    if sparql_query:
        print("\n--- GENERATED SPARQL ---")
        print(sparql_query)

    if graph_context:
        print("\n--- GRAPH RESULT ---")
        print(graph_context)

    if vector_context:
        print("\n--- DOCUMENT CONTEXT ---")
        print(vector_context)

    print("\n--- FINAL ANSWER ---")
    print(final_answer)


if __name__ == "__main__":
    main()