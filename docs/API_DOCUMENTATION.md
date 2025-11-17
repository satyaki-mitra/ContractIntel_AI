# AI Contract Risk Analyzer API Documentation

This document details the REST API endpoints for the AI Contract Risk Analyzer service.

**Base URL:** `http://<your-host>:<your-port>/api/v1` (e.g., `http://localhost:8000/api/v1`)

## Table of Contents

*   [Health Check](#health-check)
*   [Get Service Status](#get-service-status)
*   [Get Contract Categories](#get-contract-categories)
*   [Analyze Contract from File](#analyze-contract-from-file)
*   [Analyze Contract from Text](#analyze-contract-from-text)
*   [Generate PDF Report](#generate-pdf-report)
*   [Validate Contract File](#validate-contract-file)
*   [Validate Contract Text](#validate-contract-text)

---

## Health Check

Checks the basic health and availability of the API service.

### Endpoint

`GET /api/v1/health`

### Request

No body required.

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-17T15:00:00.123456",
  "models_loaded": 5,
  "services_loaded": 6,
  "memory_usage_mb": 2048.5
}
```

---

## Get Service Status

Retrieves detailed status information about the loaded models and services.

### Endpoint

`GET /api/v1/status`

### Request

No body required.

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "services": {
    "classifier": "loaded",
    "clause_extractor": "loaded",
    "risk_analyzer": "loaded",
    "llm_interpreter": "loaded",
    "negotiation_engine": "loaded",
    "summary_generator": "loaded",
    "term_analyzer": "loaded",
    "protection_checker": "loaded"
  },
  "models": {
    "legal-bert": {
      "name": "legal-bert",
      "type": "LEGAL_BERT",
      "status": "LOADED",
      "loaded_at": "2025-11-17T14:55:00.123456",
      "memory_size_mb": 400.0,
      "access_count": 10,
      "last_accessed": "2025-11-17T15:00:00.123456"
    },
    "embedding": {
      "name": "embedding",
      "type": "EMBEDDING",
      "status": "LOADED",
      "loaded_at": "2025-11-17T14:55:00.123456",
      "memory_size_mb": 100.0,
      "access_count": 8,
      "last_accessed": "2025-11-17T14:59:59.123456"
    }
  },
  "memory_usage_mb": 2048.5,
  "total_services_loaded": 8,
  "total_models_loaded": 5
}
```

---

## Get Contract Categories

Retrieves a list of contract categories that the classifier can identify.

### Endpoint

`GET /api/v1/categories`

### Request

No body required.

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "categories": [
    "employment",
    "consulting",
    "nda",
    "software",
    "service",
    "partnership",
    "lease",
    "purchase",
    "general"
  ]
}
```

---

## Analyze Contract from File

Uploads a contract file (PDF, DOCX, TXT) for analysis.

### Endpoint

`POST /api/v1/analyze/file`

### Request

**Content-Type:** `multipart/form-data`

**Form Data:**

- `file`: **(Required)** The contract file to analyze (PDF, DOCX, TXT).
- `max_clauses`: **(Optional, Integer)** Maximum number of clauses to analyze (default: `50`, min: `5`, max: `30`).
- `interpret_clauses`: **(Optional, Boolean)** Whether to generate LLM interpretations for clauses (default: `true`).
- `generate_negotiation_points`: **(Optional, Boolean)** Whether to generate negotiation points (default: `true`).
- `compare_to_market`: **(Optional, Boolean)** Whether to perform market comparison (default: `false`, currently disabled).

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "timestamp": "2025-11-17T15:01:00.123456",
  "classification": {
    "category": "employment",
    "subcategory": "executive",
    "confidence": 0.95,
    "reasoning": ["Keywords like 'executive', 'compensation', 'non-compete' found"],
    "detected_keywords": ["employment", "executive", "non-compete", "compensation"]
  },
  "clauses": [
    {
      "text": "Employee agrees to a 24-month non-compete...",
      "reference": "Clause 9.5",
      "category": "restrictive_covenants",
      "confidence": 0.98,
      "start_pos": 1200,
      "end_pos": 1350,
      "extraction_method": "semantic",
      "risk_indicators": ["non-compete", "24 months", "entire industry"],
      "risk_score": 90
    }
  ],
  "risk_analysis": {
    "overall_score": 85,
    "risk_level": "CRITICAL",
    "category_scores": {
      "restrictive_covenants": 95,
      "penalties_liability": 90,
      "compensation_benefits": 80
    },
    "risk_factors": ["Restrictive Covenants", "Penalties & Liability"],
    "detailed_findings": {
      "restrictive_covenants": [
        "Non-compete clause (Clause 9.5) is overly broad and long.",
        "Non-solicitation clause (Clause 17.6) has excessive duration."
      ]
    },
    "benchmark_comparison": {
      "overall": "✗ Significantly above market risk levels",
      "high_risk_areas": ["Restrictive Covenants", "Penalties & Liability"]
    },
    "risk_breakdown": [
      {
        "category": "Restrictive Covenants",
        "score": 95,
        "summary": "The agreement contains exceptionally broad and long-lasting non-compete...",
        "findings": ["Non-compete clause (Clause 9.5) is overly broad and long."]
      }
    ],
    "contract_type": "employment",
    "unfavorable_terms": [],
    "missing_protections": []
  },
  "unfavorable_terms": [
    {
      "term": "Risk Factor: entire industry",
      "category": "restrictive_covenants",
      "severity": "critical",
      "explanation": "Non-compete restricts the Employee from applying to any company in the same 'Industry'...",
      "risk_score": 90,
      "clause_reference": "Clause 9.5",
      "suggested_fix": "Negotiate to have this clause removed entirely...",
      "contract_type": "EMPLOYMENT",
      "specific_text": "entire industry",
      "benchmark_info": null,
      "legal_basis": "Reasonableness standard for restrictive covenants"
    }
  ],
  "missing_protections": [
    {
      "protection": "For Cause Definition",
      "importance": "critical",
      "risk_score": 25,
      "explanation": "Without a clear 'for cause' definition, termination grounds remain ambiguous...",
      "recommendation": "Add clear 'For Cause' definition including...",
      "categories": ["termination_rights"],
      "contract_type": "EMPLOYMENT",
      "suggested_language": "\"For Cause\" means: (a) gross negligence...",
      "legal_basis": "Employment protection statutes...",
      "affected_clauses": ["Clause 17.1"]
    }
  ],
  "clause_interpretations": [
    {
      "clause_reference": "Clause 9.5",
      "original_text": "Employee agrees to a 24-month non-compete...",
      "plain_english_summary": "You cannot work for or apply to any company in the same industry for 24 months after leaving.",
      "key_points": [
        "Duration: 24 months",
        "Scope: Entire industry",
        "Applies to: Applying for jobs too"
      ],
      "potential_risks": [
        "Severely limits future job opportunities.",
        "Scope is likely unenforceable."
      ],
      "favorability": "unfavorable",
      "confidence": 0.85,
      "risk_score": 90,
      "negotiation_priority": "high",
      "suggested_improvements": [
        "Reduce duration to 6-12 months.",
        "Narrow scope to direct competitors only."
      ]
    }
  ],
  "negotiation_points": [
    {
      "priority": 1,
      "category": "restrictive_covenants",
      "issue": "Extremely broad non-compete clause",
      "current_language": "Employee agrees to a 24-month non-compete...",
      "proposed_language": "Limit non-compete to 6 months and direct competitors only.",
      "rationale": "The current clause is overly broad and likely unenforceable.",
      "tactic": "limitation",
      "fallback_position": "If 6 months is not accepted, propose 12 months.",
      "estimated_difficulty": "medium",
      "legal_basis": "Reasonableness standard for restrictive covenants",
      "business_impact": "Severely restricts the Employee's ability to find future employment.",
      "counterparty_concerns": "They may argue this is necessary to protect trade secrets.",
      "timing_suggestion": "Address this early in negotiations.",
      "bargaining_chips": [
        "Offer to sign a stronger confidentiality agreement.",
        "Agree to a shorter notice period for termination."
      ]
    }
  ],
  "market_comparisons": [],
  "executive_summary": "This employment agreement is heavily skewed in favor of the Employer...",
  "metadata": {
    "text_length": 15000,
    "word_count": 2500,
    "num_clauses": 20,
    "contract_type": "EMPLOYMENT",
    "actual_category": "employment",
    "options": {
      "max_clauses": 50,
      "interpret_clauses": true,
      "generate_negotiation_points": true,
      "compare_to_market": false
    }
  },
  "pdf_available": true
}
```

### Error Response

**Status Code:** `400 Bad Request` or `500 Internal Server Error`

**Content-Type:** `application/json`

```json
{
  "error": "Analysis failed",
  "detail": "Contract text too short. Minimum 300 characters required.",
  "timestamp": "2025-11-17T15:01:01.123456"
}
```

---

## Analyze Contract from Text

Analyzes a contract provided as plain text.

### Endpoint

`POST /api/v1/analyze/text`

### Request

**Content-Type:** `application/x-www-form-urlencoded`

**Form Data:**

- `contract_text`: **(Required, String)** The full text of the contract.
- `max_clauses`: **(Optional, Integer)** Maximum number of clauses to analyze (default: `15`, min: `5`, max: `30`).
- `interpret_clauses`: **(Optional, Boolean)** Whether to generate LLM interpretations for clauses (default: `true`).
- `generate_negotiation_points`: **(Optional, Boolean)** Whether to generate negotiation points (default: `true`).
- `compare_to_market`: **(Optional, Boolean)** Whether to perform market comparison (default: `false`, currently disabled).

### Response

Same as the response for [Analyze Contract from File](#analyze-contract-from-file).

### Error Response

Same as the error response for [Analyze Contract from File](#analyze-contract-from-file).

---

## Generate PDF Report

Generates a downloadable PDF report based on the analysis result provided in the request body.

### Endpoint

`POST /api/v1/generate-pdf`

### Request

**Content-Type:** `application/json`

**Body:** The full JSON object returned by a successful `/analyze/file` or `/analyze/text` request.

```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "timestamp": "2025-11-17T15:01:00.123456",
  "classification": { ... },
  "clauses": [ ... ],
  "risk_analysis": { ... },
  "unfavorable_terms": [ ... ],
  "missing_protections": [ ... ],
  "clause_interpretations": [ ... ],
  "negotiation_points": [ ... ],
  "market_comparisons": [ ... ],
  "executive_summary": "...",
  "metadata": { ... },
  "pdf_available": true
}
```

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/pdf`

**Headers:**

- `Content-Disposition`: `attachment; filename=contract_analysis_<analysis_id>.pdf`

The response body contains the binary PDF data.

### Error Response

**Status Code:** `500 Internal Server Error`

**Content-Type:** `application/json`

```json
{
  "error": "Internal server error",
  "detail": "Failed to generate PDF: Some error message",
  "timestamp": "2025-11-17T15:02:00.123456"
}
```

---

## Validate Contract File

Validates if an uploaded file is a potentially valid contract document.

### Endpoint

`POST /api/v1/validate/file`

### Request

**Content-Type:** `multipart/form-data`

**Form Data:**

- `file`: **(Required)** The contract file to validate (PDF, DOCX, TXT).

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "valid": true,
  "message": "Contract appears valid",
  "confidence": 85.0,
  "report": {
    "scores": {
      "total": 85.0,
      "has_parties": 90.0,
      "has_date": 80.0,
      "has_terms": 90.0
    },
    "found_indicators": ["agreement", "party", "terms"],
    "found_anti_patterns": [],
    "text_statistics": {
      "length": 15000,
      "word_count": 2500,
      "line_count": 300
    }
  }
}
```

### Error Response

**Status Code:** `400 Bad Request`

**Content-Type:** `application/json`

```json
{
  "error": "Validation failed",
  "detail": "File too large. Max size: 10.0MB",
  "timestamp": "2025-11-17T15:03:00.123456"
}
```

---

## Validate Contract Text

Validates if a provided text string is a potentially valid contract.

### Endpoint

`POST /api/v1/validate/text`

### Request

**Content-Type:** `application/x-www-form-urlencoded`

**Form Data:**

- `contract_text`: **(Required, String)** The text to validate.

### Response

**Status Code:** `200 OK`

**Content-Type:** `application/json`

```json
{
  "valid": true,
  "message": "Contract appears valid",
  "confidence": 78.0,
  "report": {
    "scores": {
      "total": 78.0,
      "has_parties": 85.0,
      "has_date": 70.0,
      "has_terms": 80.0
    },
    "found_indicators": ["agreement", "party", "payment"],
    "found_anti_patterns": [],
    "text_statistics": {
      "length": 1200,
      "word_count": 200,
      "line_count": 25
    }
  }
}
```

### Error Response

**Status Code:** `400 Bad Request`

**Content-Type:** `application/json`

```json
{
  "error": "Validation failed",
  "detail": "Contract text too short. Minimum 300 characters required.",
  "timestamp": "2025-11-17T15:04:00.123456"
}
```

---

## Notes

- All timestamps are in ISO 8601 format
- All risk scores are integers from 0-100
- The API uses custom JSON serialization to handle NumPy types
- CORS is enabled for all origins in development
- Maximum file upload size is configurable via settings (default: 10MB)
- Minimum contract text length: 300 characters (configurable)
- Maximum contract text length: configurable via settings