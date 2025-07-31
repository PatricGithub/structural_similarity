# prompts.py - Structural Similarity Analysis Based on Gentner (1983)

STRUCTURAL_SIMILARITY_PROMPT = """
You are a research assistant analyzing chess player verbal responses for structural similarity based on Gentner's (1983) definition.

**IMPORTANT: The response text is in GERMAN. You must understand and analyze the German text accurately.**

**Definition of Structural Similarity (Gentner, 1983):**
Structural similarity refers to shared RELATIONSHIPS between objects in two domains, rather than shared ATTRIBUTES of the objects themselves.
- Focus on mapping of relations between objects, not attributes of objects
- Look for relational structures that apply across domains
- Example: In 3:6::2:4, the relationship "twice as great as" matters, not the individual numbers

**In Chess Context:**
- STRUCTURAL (high structural similarity): Focuses on relationships between pieces, how pieces interact, support each other, create patterns, control squares together
- NON-STRUCTURAL (low structural similarity): Focuses on individual piece attributes, isolated positions, surface features like color or location

**Verbal Response to Analyze:**
Response: "{response_text}"

**Instructions:**
Analyze the German response for structural similarity and return a JSON object with these exact keys:

{{
    "structural_similarity_score": 0,
    "analysis_category": "structural",
    "relational_expressions": ["list of relational expressions found"],
    "attribute_expressions": ["list of attribute/surface expressions found"],
    "explanation": "Detailed explanation of why this response shows structural/non-structural thinking",
    "key_evidence": ["specific phrases that demonstrate structural or non-structural focus"]
}}

**Scoring Guidelines:**
- structural_similarity_score: 0-10 scale
  - 8-10: Strong focus on piece relationships, interactions, patterns
  - 5-7: Mixed approach with some relational thinking
  - 2-4: Mostly attribute-focused with minimal relationships
  - 0-1: Pure attribute/surface description

**analysis_category values:**
- "structural": Predominantly relational thinking (score 7-10)
- "mixed": Balance of relational and attribute focus (score 4-6)
- "non-structural": Predominantly attribute/surface thinking (score 0-3)

**German Relational Expressions to Look For:**
- Piece relationships: "unterstützt", "deckt", "arbeiten zusammen", "koordiniert", "verbindet", "kontrollieren gemeinsam"
- Strategic relationships: "Bauernkette", "Figurenharmonie", "Zusammenspiel", "wechselseitig"
- Causal relationships: "führt zu", "ermöglicht", "verhindert", "blockiert", "öffnet"
- Spatial relationships: "zwischen", "verbunden mit", "in Beziehung zu"

**German Attribute/Surface Expressions:**
- Individual positions: "steht auf", "befindet sich", "ist auf Feld"
- Isolated descriptions: "weiß", "schwarz", "Turm", "Bauer" (without relational context)
- Surface features: "in der Ecke", "am Rand", "in der Mitte" (without strategic context)

Return ONLY the JSON object with no additional text.
"""