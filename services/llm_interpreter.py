"""
LLM Clause Interpreter
Generates plain-English explanations for legal clauses using LLM APIs
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_manager.llm_manager import LLMManager, LLMProvider
from services.clause_extractor import ExtractedClause
from utils.logger import ContractAnalyzerLogger, log_info, log_error


@dataclass
class ClauseInterpretation:
    """Plain-English interpretation of a legal clause"""
    clause_reference: str
    original_text: str
    plain_english_summary: str
    key_points: List[str]
    potential_risks: List[str]
    favorability: str  # "favorable", "neutral", "unfavorable"
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "clause_reference": self.clause_reference,
            "original_text": self.original_text,
            "plain_english_summary": self.plain_english_summary,
            "key_points": self.key_points,
            "potential_risks": self.potential_risks,
            "favorability": self.favorability,
            "confidence": self.confidence
        }


class LLMClauseInterpreter:
    """
    Uses LLM to generate plain-English explanations for legal clauses
    Supports multiple LLM providers with fallback
    """
    
    def __init__(self, llm_manager: LLMManager, 
                 default_provider: LLMProvider = LLMProvider.OLLAMA):
        """
        Initialize LLM interpreter
        
        Args:
            llm_manager: LLMManager instance
            default_provider: Default LLM provider to use
        """
        self.llm_manager = llm_manager
        self.default_provider = default_provider
        self.logger = ContractAnalyzerLogger.get_logger()
        
        log_info("LLMClauseInterpreter initialized", 
                default_provider=default_provider.value)
    
    @ContractAnalyzerLogger.log_execution_time("interpret_clauses")
    def interpret_clauses(self, clauses: List[ExtractedClause],
                         max_clauses: int = 10,
                         provider: Optional[LLMProvider] = None) -> List[ClauseInterpretation]:
        """
        Generate plain-English interpretations for multiple clauses
        
        Args:
            clauses: List of extracted clauses
            max_clauses: Maximum number to interpret (for cost control)
            provider: LLM provider to use (default: self.default_provider)
        
        Returns:
            List of ClauseInterpretation objects
        """
        provider = provider or self.default_provider
        
        log_info(f"Starting clause interpretation",
                num_clauses=min(len(clauses), max_clauses),
                provider=provider.value)
        
        # Prioritize clauses by risk indicators and confidence
        prioritized = self._prioritize_clauses(clauses, max_clauses)
        
        interpretations = []
        
        for clause in prioritized:
            try:
                interpretation = self._interpret_single_clause(clause, provider)
                interpretations.append(interpretation)
            except Exception as e:
                log_error(e, context={
                    "component": "LLMClauseInterpreter",
                    "operation": "interpret_single_clause",
                    "clause_reference": clause.reference
                })
                # Continue with other clauses even if one fails
                continue
        
        log_info(f"Clause interpretation complete",
                successful=len(interpretations),
                failed=len(prioritized) - len(interpretations))
        
        return interpretations
    
    def _prioritize_clauses(self, clauses: List[ExtractedClause], 
                           max_clauses: int) -> List[ExtractedClause]:
        """Prioritize clauses for interpretation (high-risk first)"""
        # Score each clause
        scored = []
        for clause in clauses:
            score = (
                len(clause.risk_indicators) * 3 +  # Risk indicators
                clause.confidence * 2 +             # Confidence
                (1 if clause.category in ['non_compete', 'termination', 'indemnification'] else 0) * 2
            )
            scored.append((clause, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [clause for clause, _ in scored[:max_clauses]]
    
    def _interpret_single_clause(self, clause: ExtractedClause,
                                 provider: LLMProvider) -> ClauseInterpretation:
        """Generate plain-English interpretation for a single clause"""
        
        # Create prompt for LLM
        prompt = self._create_interpretation_prompt(clause)
        
        # Call LLM with structured output
        schema_description = """
        {
            "plain_english_summary": "string (1-2 sentence summary in simple terms)",
            "key_points": ["string", "string", ...] (3-5 key points),
            "potential_risks": ["string", "string", ...] (2-4 potential risks),
            "favorability": "string (one of: favorable, neutral, unfavorable)"
        }
        """
        
        try:
            result = self.llm_manager.generate_structured_json(
                prompt=prompt,
                schema_description=schema_description,
                provider=provider,
                temperature=0.3,
                max_tokens=1000,
                fallback_providers=[LLMProvider.OPENAI, LLMProvider.ANTHROPIC]
            )
            
            # Parse result
            interpretation = ClauseInterpretation(
                clause_reference=clause.reference,
                original_text=clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
                plain_english_summary=result.get("plain_english_summary", "Unable to generate summary"),
                key_points=result.get("key_points", []),
                potential_risks=result.get("potential_risks", []),
                favorability=result.get("favorability", "neutral"),
                confidence=0.85  # High confidence if LLM succeeded
            )
            
            log_info(f"Clause interpreted successfully",
                    clause_reference=clause.reference,
                    favorability=interpretation.favorability)
            
            return interpretation
            
        except Exception as e:
            log_error(e, context={
                "component": "LLMClauseInterpreter",
                "operation": "_interpret_single_clause",
                "clause_reference": clause.reference
            })
            
            # Fallback to rule-based interpretation
            return self._fallback_interpretation(clause)
    
    def _create_interpretation_prompt(self, clause: ExtractedClause) -> str:
        """Create prompt for LLM interpretation"""
        
        risk_context = ""
        if clause.risk_indicators:
            risk_context = f"\nRisk indicators detected: {', '.join(clause.risk_indicators)}"
        
        prompt = f"""You are a legal expert explaining contract clauses to non-lawyers.

CLAUSE ({clause.reference} - {clause.category}):
\"\"\"{clause.text}\"\"\"
{risk_context}

Provide a plain-English interpretation suitable for someone without legal training:

1. SUMMARY: Explain what this clause means in 1-2 simple sentences
2. KEY POINTS: List 3-5 key things to understand about this clause
3. POTENTIAL RISKS: Identify 2-4 potential risks or concerns with this clause
4. FAVORABILITY: Rate as "favorable", "neutral", or "unfavorable" from the recipient's perspective

Return ONLY valid JSON. Be clear, concise, and focus on practical implications."""
        
        return prompt
    
    def _fallback_interpretation(self, clause: ExtractedClause) -> ClauseInterpretation:
        """Fallback rule-based interpretation if LLM fails"""
        
        category_summaries = {
            "compensation": "This clause defines how and when you will be paid for your work.",
            "termination": "This clause specifies the conditions under which the agreement can be ended.",
            "non_compete": "This clause restricts your ability to work for competitors or start competing businesses.",
            "confidentiality": "This clause requires you to keep certain information secret.",
            "indemnification": "This clause makes you financially responsible for certain types of claims or losses.",
            "intellectual_property": "This clause determines who owns the work you create.",
            "liability": "This clause limits or defines financial responsibility for damages.",
            "warranty": "This clause contains promises or guarantees about performance or quality.",
            "dispute_resolution": "This clause specifies how disagreements will be resolved.",
        }
        
        summary = category_summaries.get(
            clause.category,
            f"This is a {clause.category} clause that defines specific obligations or rights."
        )
        
        key_points = [
            f"This is classified as a {clause.category} clause",
            f"Located at {clause.reference} in the contract"
        ]
        
        if clause.risk_indicators:
            key_points.append(f"Contains risk indicators: {', '.join(clause.risk_indicators[:3])}")
        
        potential_risks = clause.risk_indicators[:4] if clause.risk_indicators else [
            "Unable to assess risks without LLM analysis"
        ]
        
        favorability = "unfavorable" if len(clause.risk_indicators) >= 2 else "neutral"
        
        return ClauseInterpretation(
            clause_reference=clause.reference,
            original_text=clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
            plain_english_summary=summary,
            key_points=key_points,
            potential_risks=potential_risks,
            favorability=favorability,
            confidence=0.50  # Lower confidence for fallback
        )
    
    def interpret_specific_clause(self, clause_text: str,
                                  clause_reference: str = "Unknown",
                                  category: str = "general",
                                  provider: Optional[LLMProvider] = None) -> ClauseInterpretation:
        """
        Interpret a specific clause text directly
        
        Args:
            clause_text: The clause text to interpret
            clause_reference: Reference identifier
            category: Clause category
            provider: LLM provider
        
        Returns:
            ClauseInterpretation object
        """
        # Create temporary ExtractedClause
        temp_clause = ExtractedClause(
            text=clause_text,
            reference=clause_reference,
            category=category,
            confidence=1.0,
            start_pos=0,
            end_pos=len(clause_text),
            extraction_method="manual",
            risk_indicators=[],
            legal_bert_score=0.0
        )
        
        return self._interpret_single_clause(temp_clause, provider or self.default_provider)
    
    def batch_interpret(self, clauses: List[ExtractedClause],
                       provider: Optional[LLMProvider] = None) -> List[ClauseInterpretation]:
        """
        Batch interpretation with progress tracking
        Alias for interpret_clauses with all clauses
        """
        return self.interpret_clauses(
            clauses=clauses,
            max_clauses=len(clauses),
            provider=provider
        )
    
    def get_unfavorable_interpretations(self, 
                                       interpretations: List[ClauseInterpretation]) -> List[ClauseInterpretation]:
        """Filter to only unfavorable clause interpretations"""
        unfavorable = [i for i in interpretations if i.favorability == "unfavorable"]
        
        log_info(f"Found {len(unfavorable)} unfavorable interpretations")
        
        return unfavorable
    
    def get_high_risk_interpretations(self,
                                     interpretations: List[ClauseInterpretation],
                                     min_risk_count: int = 2) -> List[ClauseInterpretation]:
        """Filter to interpretations with multiple risks"""
        high_risk = [i for i in interpretations if len(i.potential_risks) >= min_risk_count]
        
        log_info(f"Found {len(high_risk)} high-risk interpretations")
        
        return high_risk