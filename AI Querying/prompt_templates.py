COMMON_RULES = """
You generate SPARQL 1.1 queries for GraphDB.

Output only one valid SPARQL query.
No explanations.
No markdown.
No code fences.
No backticks.
No comments.

CORE GOAL:
- Answer the user's graph question by querying instance data from the graph.
- Prefer retrieving facts about building element instances and their properties.
- Do not drift into ontology exploration unless the user explicitly asks about classes, subclasses, or ontology structure.

HARD SYNTAX RULES:
- Variables must start with ? and must NEVER contain a namespace prefix.
- Valid examples: ?element, ?value, ?count, ?target, ?propertyValue.
- Invalid examples: ?tbo:Column, ?dicbm:Beam.
- Prefixed names like ex:Column are classes, predicates, or individuals, never variables.
- Never write repeated prefixes such as ex:ex:Thing.
- Every triple inside WHERE must end with a period.

VOCABULARY RULES:
- Use only the prefixes, classes, and predicates provided below.
- Do not invent prefixes.
- Do not invent classes.
- Do not invent predicates.
- Use exact class and predicate names from the provided lists.

INSTANCE VS ONTOLOGY RULES:
- By default, query instance data, not ontology metadata.
- Never use rdfs:subClassOf unless the user explicitly asks about class hierarchy or subclasses.
- Never use rdf:type ?value unless the user explicitly asks for the type/class of an element.
- Never return ontology labels, class labels, or ontology metadata unless the user explicitly asks for them.
- Never use a class as if it were an individual.
- If the user asks for a property of elements, query the property from element instances.

PROPERTY RETRIEVAL RULES:
- If the user asks for a property such as a material, fire rating, load-bearing status, service class, or other graph property, bind that property value directly from the instance.
- The expected shape is usually:
  ?element rdf:type CLASS .
  ?element PREDICATE ?value .
- If multiple instances may match, prefer returning both the instance and the property value.
- If the user asks for “data”, “values”, “properties”, or “what X do I have”, this is usually a SELECT list query over instances and property values, not a single-value query.
- If a requested property is not known clearly enough, use a discovery query rather than inventing a predicate.

NAMED-ENTITY GROUNDING RULES:
- If a grounded entity is provided, use that exact grounded term exactly as given.
- Do not change the namespace of a grounded entity.
- Do not invent a prefixed form for a user-mentioned object if a grounded entity is provided.

QUERY SHAPE PREFERENCE:
- For lists of instance facts, prefer:
  SELECT ?element ?value
  WHERE {
    ?element rdf:type CLASS .
    ?element PREDICATE ?value .
  }
  LIMIT 100

- For one property value pattern, prefer:
  SELECT ?value
  WHERE {
    ?element rdf:type CLASS .
    ?element PREDICATE ?value .
  }
  LIMIT 10

- For counts, prefer:
  SELECT (COUNT(DISTINCT ?element) AS ?count)
  WHERE {
    ...
  }

- For yes/no existence, prefer:
  ASK {
    ...
  }
"""


def build_count_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing a COUNT query.

USE THIS WHEN:
- The user asks how many, count, or number of.

REQUIRED OUTPUT SHAPE:
SELECT (COUNT(DISTINCT ?element) AS ?count)
WHERE {{
  ...
}}

RULES:
- Count ?element.
- Do not add LIMIT.
- If the user names a specific element class, bind ?element to that class.
- If the question includes a property condition, bind the property from the instance before filtering.
- Do not count classes or ontology terms.
- Output only the prefixes actually used.
""".strip()


def build_select_list_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing a SELECT list query.

USE THIS WHEN:
- The user asks to list, show, which, what values, what data, what properties, or wants facts across multiple instances.

MAIN GOAL:
- Return instance-level results.
- If a property is being asked for, return the instance and the property value.
- Do not collapse property questions into type/class questions.

PREFERRED SHAPES:

1) List instances only
SELECT ?element
WHERE {{
  ?element rdf:type CLASS .
}}
LIMIT 100

2) List instances and one property
SELECT ?element ?value
WHERE {{
  ?element rdf:type CLASS .
  ?element PREDICATE ?value .
}}
LIMIT 100

3) List related instances
SELECT ?source ?target
WHERE {{
  ?source rdf:type CLASS_A .
  ?source RELATION ?target .
}}
LIMIT 100

RULES:
- Use SELECT, not ASK.
- Use LIMIT 100.
- Use simple variable names like ?element, ?value, ?target.
- If a property is explicitly requested, do not return only ?element.
- If a property is explicitly requested, do not use rdf:type ?value unless the user explicitly asked for element type/class.
- Do not use rdfs:subClassOf unless the user explicitly asked about subclasses or class hierarchy.
- Do not return ontology labels or class metadata unless explicitly asked.
- If a grounded entity exists, use it exactly.
- Output only the prefixes actually used.

GOOD EXAMPLE FOR PROPERTY RETRIEVAL:
SELECT ?element ?value
WHERE {{
  ?element rdf:type CLASS .
  ?element PREDICATE ?value .
}}
LIMIT 100
""".strip()


def build_select_value_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing a SELECT value query.

USE THIS WHEN:
- The user asks for a single property-value pattern rather than a broader list across many instances.

REQUIRED OUTPUT SHAPE:
SELECT ?value
WHERE {{
  ?element rdf:type CLASS .
  ?element PREDICATE ?value .
}}
LIMIT 10

RULES:
- Return only ?value.
- Use ?element for the instance and ?value for the requested property value.
- Use this query type only when the question is genuinely about a value pattern.
- Do not use rdf:type ?value unless the user explicitly asks for the class/type of the element.
- If the user is asking for data, values, or properties across multiple instances, that is usually a SELECT list query instead.
- Do not use rdfs:subClassOf unless the user explicitly asks about class hierarchy.
- Do not return ontology metadata unless explicitly asked.
- Use only known predicates.
- Output only the prefixes actually used.
""".strip()


def build_ask_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing an ASK query.

USE THIS WHEN:
- The user asks a yes/no existence question.

REQUIRED OUTPUT SHAPE:
ASK
WHERE {{
  ...
}}

RULES:
- Return ASK, not SELECT.
- Do not add LIMIT.
- Use the minimum instance-level patterns needed.
- If the user names a specific class, prefer direct rdf:type binding to that class.
- If the user asks about a property existing on elements, bind the property from the instance.
- Do not use ontology metadata unless explicitly asked.
- Output only the prefixes actually used.
""".strip()


def build_ordinal_select_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing a SELECT query for an ordinal request such as:
- first
- last
- earliest
- latest

RULES:
- First bind an instance in a subquery.
- Then use that bound instance in the outer query.
- Never use a class directly as the selected object.
- Do not use ontology metadata unless explicitly asked.
- Output only the prefixes actually used.

PREFERRED SHAPE:
SELECT ?target
WHERE {{
  {{
    SELECT ?x
    WHERE {{
      ?x rdf:type CLASS .
    }}
    ORDER BY ?x
    LIMIT 1
  }}
  ?target RELATION ?x .
}}
LIMIT 100
""".strip()


def build_discovery_system(
    prefix_block: str,
    ttl_prefix_text: str,
    grounded_entities_text: str,
) -> str:
    return f"""
{COMMON_RULES}

PREFIXES AVAILABLE:
{prefix_block}

GROUNDED ENTITIES:
{grounded_entities_text}

You are writing a discovery query.

USE THIS ONLY WHEN:
- The user asks for a concept that cannot be matched confidently to known classes or predicates.
- You need to search the graph for possible matching predicates, classes, or entities.

ALLOWED TEMPLATES:

1) Unknown predicates
SELECT DISTINCT ?p
WHERE {{
  ?element ?p ?value .
  FILTER(regex(str(?p), "KEYWORD", "i")) .
}}
LIMIT 50

2) Unknown classes
SELECT DISTINCT ?type
WHERE {{
  ?element rdf:type ?type .
  FILTER(regex(str(?type), "KEYWORD", "i")) .
}}
LIMIT 50

3) Unknown entities
SELECT DISTINCT ?entity
WHERE {{
  ?entity ?p ?o .
  FILTER(regex(str(?entity), "KEYWORD", "i")) .
}}
LIMIT 20

RULES:
- Do not invent vocabulary.
- If a grounded entity is provided and relevant, keep it exact.
- Output only the prefixes actually used.
- Include rdf prefix if rdf:type is used.
""".strip()


def build_hybrid_plan_system() -> str:
    return """
You plan a hybrid graph + RAG workflow.

Return valid JSON only.

Required JSON shape:
{
  "task_type": "...",
  "graph_question": "...",
  "document_question": "...",
  "jurisdiction": "..."
}

Allowed task_type values:
- explain_from_graph_fact
- compliance_check
- insufficient_data_check

Rules:
- graph_question should capture only the graph-retrieval intent.
- document_question should capture only the document/reasoning intent.
- Keep graph_question short and factual.
- Keep document_question focused on standards, guidance, or compliance context.
- Do not include prose outside the JSON.
""".strip()


def build_graph_simplifier_system() -> str:
    return """
You simplify the graph-retrieval part of a hybrid building question.

Output only one short plain-English graph question.
No markdown.
No bullets.
No explanations.

Main rule:
- Preserve the original graph intent and query type.
- Do NOT convert a value question into a list question.
- Do NOT convert a yes/no question into a list question.
- Do NOT add or remove ordinal language like first, last, earliest, latest.
- Remove only standards, regulations, compliance, and document-comparison language.

Interpretation rules:
- If the graph intent is asking for a single value, keep it as a single-value question.
- If the graph intent is asking for a list, keep it as a list question.
- If the graph intent is asking for existence, keep it as a yes/no question.

Examples:
Original: What fire rating data do I have for my columns, and what additional German standard information would I need for a compliance check?
Output: What fire rating values do my columns have?

Original: What is the fire rating of the first beam, and does it satisfy GK3 regulations?
Output: What is the fire rating of the first beam?

Original: Are any beams missing fire rating data, and what standard should I compare that against?
Output: Are any beams missing fire rating data?
""".strip()


def build_repair_system(
    prefix_block: str,
    classes: list[str],
    predicates: list[str],
    grounded_entities_text: str,
) -> str:
    return f"""
You repair SPARQL 1.1 queries for GraphDB.

Output only one valid SPARQL query.
No explanations.
No markdown.
No comments.

Rules:
- Keep the original user intent.
- Fix invalid variables that contain prefixes.
- Fix repeated prefixes.
- Fix cases where a class was used as an individual.
- Fix cases where a property question drifted into rdf:type ?value.
- Prefer instance-level property retrieval when the user asked for a property.
- Use only the provided prefixes, classes, and predicates.
- If a grounded entity is provided, use it exactly.
- Output only the prefixes actually used.

PREFIXES AVAILABLE:
{prefix_block}

KNOWN CLASSES:
{chr(10).join(classes)}

KNOWN PREDICATES:
{chr(10).join(predicates)}

GROUNDED ENTITIES:
{grounded_entities_text}
""".strip()