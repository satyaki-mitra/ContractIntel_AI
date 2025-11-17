# DEPENDENCIES
import sys
import numpy as np
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))




@dataclass
class ExtractedClause:
    """
    Extracted clause with comprehensive metadata
    """
    text              : str
    reference         : str    # e.g., "Section 5.2", "Clause 11.1"
    category          : str    # e.g., "termination", "compensation", "indemnification"
    confidence        : float  # 0.0-1.0
    start_pos         : int
    end_pos           : int
    extraction_method : str    # "structural", "semantic", "hybrid"
    risk_indicators   : List[str]            = field(default_factory = list)
    embeddings        : Optional[np.ndarray] = None
    subclauses        : List[str]            = field(default_factory = list)
    legal_bert_score  : float                = 0.0
    risk_score        : float                = 0.0  
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        """
        return {"text"              : self.text,
                "reference"         : self.reference,
                "category"          : self.category,
                "confidence"        : round(self.confidence, 3),
                "start_pos"         : self.start_pos,
                "end_pos"           : self.end_pos,
                "extraction_method" : self.extraction_method,
                "risk_indicators"   : self.risk_indicators,
                "subclauses"        : self.subclauses,
                "legal_bert_score"  : round(self.legal_bert_score, 3),
                "risk_score"        : round(self.risk_score, 3),
               }


@dataclass
class UnfavorableTerm:
    """
    Detected unfavorable term with comprehensive risk analysis
    """
    term             : str
    category         : str    # Risk category (e.g., "restrictive_covenants")
    severity         : str    # "critical", "high", "medium", "low"
    explanation      : str
    risk_score       : float  # 0-100 risk score
    clause_reference : Optional[str] = None
    suggested_fix    : Optional[str] = None
    contract_type    : Optional[str] = None
    specific_text    : Optional[str] = None
    benchmark_info   : Optional[str] = None  # Industry benchmark comparison
    legal_basis      : Optional[str] = None  # Legal principle violated
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary
        """
        return {"term"             : self.term,
                "category"         : self.category,
                "severity"         : self.severity,
                "explanation"      : self.explanation,
                "risk_score"       : round(self.risk_score, 2),
                "clause_reference" : self.clause_reference,
                "suggested_fix"    : self.suggested_fix,
                "contract_type"    : self.contract_type,
                "specific_text"    : self.specific_text,
                "benchmark_info"   : self.benchmark_info,
                "legal_basis"      : self.legal_basis,
               }


@dataclass
class ClauseInterpretation:
    """
    LLM interpretation of a clause with comprehensive analysis
    """
    clause_reference       : str  
    original_text          : str  
    plain_english_summary  : str
    key_points             : List[str]
    potential_risks        : List[str]
    suggested_improvements : List[str]
    favorability           : str = "neutral"  
    confidence_score       : float = 0.0
    risk_level             : str = "unknown"
    negotiation_priority   : str = "medium"  
    legal_precedents       : List[str]      = field(default_factory = list)
    negotiation_leverage   : List[str]      = field(default_factory = list)
    market_comparison      : Optional[str]  = None
    risk_score             : float          = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
       
        return {"clause_reference"       : self.clause_reference,
                "original_text"          : self.original_text,
                "plain_english_summary"  : self.plain_english_summary,
                "key_points"             : self.key_points,
                "potential_risks"        : self.potential_risks,
                "suggested_improvements" : self.suggested_improvements,
                "favorability"           : self.favorability,
                "confidence_score"       : round(self.confidence_score, 3),
                "risk_level"             : self.risk_level,
                "negotiation_priority"   : self.negotiation_priority,
                "legal_precedents"       : self.legal_precedents,
                "negotiation_leverage"   : self.negotiation_leverage,
                "market_comparison"      : self.market_comparison,
                "risk_score"             : round(self.risk_score, 3),
               }


@dataclass
class MissingProtection:
    """
    Missing protection item with comprehensive risk analysis
    """
    protection_id      : str    # Internal identifier
    protection         : str
    importance         : str    # "critical", "high", "medium", "low"
    risk_score         : float  # 0-100 from risk_rules
    explanation        : str
    recommendation     : str
    categories         : List[str]
    contract_type      : Optional[str]       = None
    suggested_language : Optional[str]       = None
    legal_basis        : Optional[str]       = None
    affected_clauses   : Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """
        Convert to dictionary
        """
        return {"protection_id"      : self.protection_id,  
                "protection"         : self.protection,
                "importance"         : self.importance,
                "risk_score"         : round(self.risk_score, 2),
                "explanation"        : self.explanation,
                "recommendation"     : self.recommendation,
                "categories"         : self.categories,
                "contract_type"      : self.contract_type,
                "suggested_language" : self.suggested_language,
                "legal_basis"        : self.legal_basis,
                "affected_clauses"   : self.affected_clauses or [],
               }


@dataclass
class ContractCategory:
    """
    Contract classification result with metadata
    """
    category               : str
    subcategory            : Optional[str]
    confidence             : float
    reasoning              : List[str]
    detected_keywords      : List[str]
    alternative_categories : List[Tuple[str, float]] = None  # (category, confidence) pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        """
        return {"category"               : self.category,
                "subcategory"            : self.subcategory,
                "confidence"             : round(self.confidence, 3),
                "reasoning"              : self.reasoning,
                "detected_keywords"      : self.detected_keywords,
                "alternative_categories" : [{"category": cat, "confidence": round(conf, 3)} for cat, conf in (self.alternative_categories or [])]
               }



@dataclass
class RiskBreakdownItem:
    """
    Individual risk category breakdown
    """
    category : str
    score    : int  # 0-100
    summary  : str
    findings : List[str] = field(default_factory = list)
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"category" : self.category,
                "score"    : self.score,
                "summary"  : self.summary,
                "findings" : self.findings,
               }


@dataclass
class RiskScore:
    """
    Comprehensive risk score with detailed breakdown
    """
    overall_score        : int  # 0-100 
    risk_level           : str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category_scores      : Dict[str, int] 
    risk_factors         : List[str] 
    detailed_findings    : Dict[str, List[str]] 
    benchmark_comparison : Dict[str, str] 
    risk_breakdown       : List[RiskBreakdownItem]
    contract_type        : str
    unfavorable_terms    : List[Dict] 
    missing_protections  : List[Dict] 
    high_risk_clauses    : List[Dict]               = field(default_factory = list) 
    explanation          : str                      = "" 
    recommendations      : List[str]                = field(default_factory = list) 
    analysis_timestamp   : Optional[str]            = None 
    contract_subtype     : Optional[str]            = None 
    contract_metadata    : Optional[Dict[str, Any]] = field(default_factory = dict) 
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        """
        return {"overall_score"        : self.overall_score,
                "risk_level"           : self.risk_level,
                "category_scores"      : self.category_scores,
                "risk_factors"         : self.risk_factors, 
                "detailed_findings"    : self.detailed_findings, 
                "benchmark_comparison" : self.benchmark_comparison, 
                "risk_breakdown"       : [item.to_dict() for item in self.risk_breakdown],
                "contract_type"        : self.contract_type,
                "unfavorable_terms"    : self.unfavorable_terms, 
                "missing_protections"  : self.missing_protections, 
                "high_risk_clauses"    : self.high_risk_clauses,
                "explanation"          : self.explanation,
                "recommendations"      : self.recommendations,
                "analysis_timestamp"   : self.analysis_timestamp,
                "contract_subtype"     : self.contract_subtype,
                "contract_metadata"    : self.contract_metadata,
               }


@dataclass
class RiskInterpretation:
    """
    Comprehensive risk interpretation with LLM-enhanced explanations
    """
    overall_risk_explanation : str
    key_concerns             : List[str]
    negotiation_strategy     : str
    market_comparison        : str
    clause_interpretations   : List[ClauseInterpretation]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"overall_risk_explanation" : self.overall_risk_explanation,
                "key_concerns"             : self.key_concerns,
                "negotiation_strategy"     : self.negotiation_strategy,
                "market_comparison"        : self.market_comparison,
                "clause_interpretations"   : [ci.to_dict() for ci in self.clause_interpretations],
               }


class NegotiationTactic(Enum):
    """
    Types of negotiation tactics
    """
    REMOVAL       = "removal"
    MODIFICATION  = "modification" 
    ADDITION      = "addition"
    LIMITATION    = "limitation"
    MUTUALIZATION = "mutualization"
    CLARIFICATION = "clarification"
    

@dataclass
class NegotiationPoint:
    """
    Negotiation talking point with strategic context
    """
    priority              : int  # 1 = highest, 5 = lowest
    category              : str
    issue                 : str
    current_language      : str
    proposed_language     : str
    rationale             : str
    tactic                : NegotiationTactic
    fallback_position     : Optional[str] = None
    estimated_difficulty  : str           = "medium"  # "easy", "medium", "hard"
    legal_basis           : Optional[str] = None
    business_impact       : Optional[str] = None
    counterparty_concerns : Optional[str] = None
    timing_suggestion     : Optional[str] = None
    bargaining_chips      : List[str]     = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"priority"              : self.priority,
                "category"              : self.category,
                "issue"                 : self.issue,
                "current_language"      : self.current_language,
                "proposed_language"     : self.proposed_language,
                "rationale"             : self.rationale,
                "tactic"                : self.tactic.value,
                "fallback_position"     : self.fallback_position,
                "estimated_difficulty"  : self.estimated_difficulty,
                "legal_basis"           : self.legal_basis,
                "business_impact"       : self.business_impact,
                "counterparty_concerns" : self.counterparty_concerns,
                "timing_suggestion"     : self.timing_suggestion,
                "bargaining_chips"      : self.bargaining_chips or [],
               }


@dataclass
class NegotiationPlaybook:
    """
    Comprehensive negotiation strategy
    """
    overall_strategy     : str
    critical_points      : List[NegotiationPoint]
    walk_away_items      : List[str]
    concession_items     : List[str]
    timing_guidance      : str
    risk_mitigation_plan : str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"overall_strategy"     : self.overall_strategy,
                "critical_points"      : [point.to_dict() for point in self.critical_points],
                "walk_away_items"      : self.walk_away_items,
                "concession_items"     : self.concession_items,
                "timing_guidance"      : self.timing_guidance,
                "risk_mitigation_plan" : self.risk_mitigation_plan,
               }


@dataclass
class SummaryContext:
    """
    Context data for comprehensive summary generation
    """
    contract_type         : str
    risk_score            : int
    risk_level            : str
    category_scores       : Dict[str, int]
    unfavorable_terms     : List[Dict]
    missing_protections   : List[Dict]
    clauses               : List
    key_findings          : List[str]
    risk_interpretation   : Optional[RiskInterpretation]  = None
    negotiation_playbook  : Optional[NegotiationPlaybook] = None
    contract_text_preview : Optional[str]                 = None
    contract_metadata     : Optional[Dict[str, Any]]      = None


@dataclass 
class ModelInfo:
    """
    Model metadata and state
    """
    name           : str
    type           : str  # "legal-bert", "embedding", "tokenizer", "classifier"
    status         : str  # "not_loaded", "loading", "loaded", "error"
    model          : Optional[Any]      = None
    tokenizer      : Optional[Any]      = None
    loaded_at      : Optional[str]      = None
    error_message  : Optional[str]      = None
    memory_size_mb : float              = 0.0
    access_count   : int                = 0
    last_accessed  : Optional[str]      = None
    metadata       : Dict[str, Any]     = field(default_factory = dict)
    

    def mark_accessed(self):
        """
        Update access statistics
        """
        self.access_count += 1
        # Simple timestamp 
        self.last_accessed = "now"
    

    def get_age_seconds(self) -> float:
        """
        Get seconds since last access (simplified)
        """
        return 0.0 if not self.last_accessed else 3600.0