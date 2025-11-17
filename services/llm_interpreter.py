# DEPENDENCIES
import sys
import json
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.risk_rules import RiskRules
from config.risk_rules import ContractType
from utils.logger import ContractAnalyzerLogger
from model_manager.llm_manager import LLMManager
from services.data_models import UnfavorableTerm
from model_manager.llm_manager import LLMProvider
from services.data_models import RiskInterpretation
from services.data_models import ClauseInterpretation
from services.clause_extractor import ExtractedClause
from services.protection_checker import MissingProtection


class LLMClauseInterpreter:
    """
    Uses LLM to generate plain-English explanations for legal clauses and integrated with RiskAnalyzer results and RiskRules framework
    """
    def __init__(self, llm_manager: LLMManager, default_provider: LLMProvider = LLMProvider.OLLAMA):
        """
        Initialize LLM interpreter
        
        Arguments:
        ----------
            llm_manager      { LLMManager }  : LLMManager instance
            default_provider { LLMProvider } : Default LLM provider to use
        """
        self.llm_manager      = llm_manager
        self.default_provider = default_provider
        self.risk_rules       = RiskRules()
        self.logger           = ContractAnalyzerLogger.get_logger()
        
        log_info("LLMClauseInterpreter initialized", default_provider = default_provider.value)
    

    # Interpret with full risk context
    @ContractAnalyzerLogger.log_execution_time("interpret_with_risk_context")
    def interpret_with_risk_context(self, clauses: List[ExtractedClause], unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection],
                                    contract_type: ContractType, overall_risk_score: int, max_clauses: int = 50, provider: Optional[LLMProvider] = None) -> RiskInterpretation:
        """
        Generate comprehensive risk interpretation with full context
        
        Arguments:
        ----------
            clauses              { list }         : Extracted clauses with risk scores
            
            unfavorable_terms    { list }         : Detected unfavorable terms
            
            missing_protections  { list }         : Missing critical protections
            
            contract_type        { ContractType } : Type of contract for context
            
            overall_risk_score   { int }          : Overall risk score (0-100)
            
            max_clauses          { int }          : Maximum clauses to interpret
            
            provider             { LLMProvider }  : LLM provider to use
        
        Returns:
        --------
                  { RiskInterpretation }          : Comprehensive RiskInterpretation with explanations
        """
        provider = provider or self.default_provider
        
        log_info("Starting comprehensive risk interpretation",
                 contract_type           = contract_type.value,
                 overall_risk_score      = overall_risk_score,
                 num_clauses             = len(clauses),
                 num_unfavorable_terms   = len(unfavorable_terms),
                 num_missing_protections = len(missing_protections),
                )
        
        # Interpret key clauses with risk context
        clause_interpretations = self.interpret_clauses(clauses     = clauses,
                                                        max_clauses = max_clauses,
                                                        provider    = provider,
                                                       )
                                                    
        # Generate overall risk explanation
        overall_explanation    = self._generate_overall_risk_explanation(overall_risk_score  = overall_risk_score,
                                                                         contract_type       = contract_type,
                                                                         unfavorable_terms   = unfavorable_terms,
                                                                         missing_protections = missing_protections,
                                                                         provider            = provider,
                                                                        )
                                                                    
        # Extract key concerns
        key_concerns           = self._extract_key_concerns(unfavorable_terms      = unfavorable_terms,
                                                            missing_protections    = missing_protections,
                                                            clause_interpretations = clause_interpretations,
                                                           )
        
        # Generate negotiation strategy
        negotiation_strategy   = self._generate_negotiation_strategy(contract_type       = contract_type,
                                                                     unfavorable_terms   = unfavorable_terms,
                                                                     missing_protections = missing_protections,
                                                                     overall_risk_score  = overall_risk_score,
                                                                     provider            = provider,
                                                                    )
        
        # Market comparison
        market_comparison      = self._generate_market_comparison(contract_type      = contract_type,
                                                                  overall_risk_score = overall_risk_score,
                                                                  provider           = provider,
                                                                 )
        
        interpretation         = RiskInterpretation(overall_risk_explanation = overall_explanation,
                                                    key_concerns             = key_concerns,
                                                    negotiation_strategy     = negotiation_strategy,
                                                    market_comparison        = market_comparison,
                                                    clause_interpretations   = clause_interpretations,
                                                   )
                                                
        log_info("Comprehensive risk interpretation complete")
        
        return interpretation


    @ContractAnalyzerLogger.log_execution_time("interpret_clauses")
    def interpret_clauses(self, clauses: List[ExtractedClause], max_clauses: int = 50, provider: Optional[LLMProvider] = None) -> List[ClauseInterpretation]:
        """
        Generate plain-English interpretations for multiple clauses
        
        Arguments:
        ----------
            clauses       { list }     : List of extracted clauses
           
            max_clauses    { int }     : Maximum number to interpret (for cost control)
           
            provider   { LLMProvider } : LLM provider to use (default: self.default_provider)
        
        Returns:
        --------
                   { list }            : List of ClauseInterpretation objects
        """
        provider = provider or self.default_provider
        
        log_info(f"Starting clause interpretation", num_clauses = min(len(clauses), max_clauses), provider = provider.value)
        
        # Prioritize clauses by risk indicators and confidence
        prioritized     = self._prioritize_clauses(clauses, max_clauses)
          
        interpretations = list()
        
        for clause in prioritized:
            try:
                interpretation = self._interpret_single_clause(clause, provider)
                interpretations.append(interpretation)

            except Exception as e:
                log_error(e, context = {"component": "LLMClauseInterpreter", "operation": "interpret_single_clause", "clause_reference": clause.reference})
                # Continue with other clauses even if one fails
                continue
        
        log_info(f"Clause interpretation complete", successful = len(interpretations), failed = len(prioritized) - len(interpretations))
        
        return interpretations

    
    def _prioritize_clauses(self, clauses: List[ExtractedClause], max_clauses: int) -> List[ExtractedClause]:
        """
        Prioritize clauses for interpretation (high-risk first)
        """
        # Scoring with risk_score
        scored = list()

        for clause in clauses:
            # Base score from original logic
            base_score       = (len(clause.risk_indicators) * 3 +   # Risk indicators
                                clause.confidence * 2 +             # Confidence
                                (1 if clause.category in ['non_compete', 'termination', 'indemnification'] else 0) * 2
                               )
            
            # Add risk_score if available (from RiskAnalyzer)
            risk_score_boost = getattr(clause, 'risk_score', 0) / 10
            total_score      = base_score + risk_score_boost
            
            scored.append((clause, total_score))
        
        # Sort by score (descending)
        scored.sort(key = lambda x: x[1], reverse = True)
        
        return [clause for clause, _ in scored[:max_clauses]]

    
    def _interpret_single_clause(self, clause: ExtractedClause, provider: LLMProvider) -> ClauseInterpretation:
        """
        Generate plain-English interpretation for a single clause
        """
        # Create enhanced prompt with risk context
        prompt             = self._create_interpretation_prompt(clause)
        
        # Call LLM with structured output
        schema_description = """
                                {
                                    "plain_english_summary": "string (1-2 sentence summary in simple terms)",
                                    "key_points": ["string", "string", ...] (3-5 key points),
                                    "potential_risks": ["string", "string", ...] (2-4 potential risks),
                                    "favorability": "string (one of: favorable, neutral, unfavorable)",
                                    "suggested_improvements": ["string", "string", ...] (2-3 improvement suggestions)
                                }
                             """
        
        try:
            result               = self.llm_manager.generate_structured_json(prompt             = prompt,
                                                                             schema_description = schema_description,
                                                                             provider           = provider,
                                                                             temperature        = 0.3,
                                                                             max_tokens         = 1200,
                                                                             fallback_providers = [LLMProvider.OPENAI, LLMProvider.ANTHROPIC],
                                                                            )
            
            # Calculate negotiation priority
            negotiation_priority = self._calculate_negotiation_priority(favorability    = result.get("favorability", "neutral"),
                                                                        risk_indicators = clause.risk_indicators,
                                                                        risk_score      = getattr(clause, 'risk_score', 0),
                                                                       )
            
            # Parse result
            interpretation       = ClauseInterpretation(clause_reference       = clause.reference,
                                                        original_text          = clause.text[:500] + "..." if len(clause.text) > 500 else clause.text,
                                                        plain_english_summary  = result.get("plain_english_summary", "Unable to generate summary"),
                                                        key_points             = result.get("key_points", []),
                                                        potential_risks        = result.get("potential_risks", []),
                                                        favorability           = result.get("favorability", "neutral"),
                                                        confidence_score       = 0.85,  # High confidence if LLM succeeded
                                                        risk_score             = getattr(clause, 'risk_score', 0),
                                                        negotiation_priority   = negotiation_priority,
                                                        suggested_improvements = result.get("suggested_improvements", []),
                                                       )
            
            log_info(f"Clause interpreted successfully",
                     clause_reference     = clause.reference,
                     favorability         = interpretation.favorability,
                     negotiation_priority = negotiation_priority,
                    )
            
            return interpretation
            
        except Exception as e:
            log_error(e, context = {"component": "LLMClauseInterpreter", "operation": "_interpret_single_clause", "clause_reference": clause.reference})
            
            # Enhanced fallback with risk context
            return self._fallback_interpretation(clause)
    

    def _create_interpretation_prompt(self, clause: ExtractedClause) -> str:
        """
        Create concise prompt for clause interpretation
        """
        risk_context = ""

        if clause.risk_indicators:
            risk_context = f"\nRisk Keywords: {', '.join(clause.risk_indicators[:3])}"
        
        risk_score_context = ""

        if hasattr(clause, 'risk_score'):
            if (clause.risk_score >= 70):
                risk_level = "CRITICAL RISK"

            elif (clause.risk_score >= 50):
                risk_level = "HIGH RISK"

            else:
                risk_level = "Moderate risk"
            
            risk_score_context = f"\nRisk Level: {risk_level} ({clause.risk_score}/100)"
        
        prompt = f"""
                     Explain this legal clause in plain English.

                     CLAUSE: {clause.reference} - {clause.category.replace('_', ' ').title()}{risk_score_context}{risk_context}

                     TEXT: "{clause.text}..."

                     Provide:
                     1. SUMMARY: 1-2 sentences explaining what this means
                     2. KEY_POINTS: 3 bullet points of what to know
                     3. POTENTIAL_RISKS: 2-3 specific risks or concerns
                     4. FAVORABILITY: "favorable", "neutral", or "unfavorable"
                     5. IMPROVEMENTS: 2 specific suggestions to fix this

                     Keep each section CONCISE. Total response should be ~150 words.

                     Return ONLY valid JSON:
                     {{
                        "plain_english_summary": "...",
                        "key_points": ["...", "...", "..."],
                        "potential_risks": ["...", "..."],
                        "favorability": "unfavorable",
                        "suggested_improvements": ["...", "..."]
                     }}
                  """
        
        return prompt
    

    def _calculate_negotiation_priority(self, favorability: str, risk_indicators: List[str], risk_score: float) -> str:
        """
        Calculate negotiation priority based on multiple factors
        """
        if (favorability == "unfavorable") and ((len(risk_indicators) >= 3) or (risk_score >= 70)):
            return "high"

        elif (favorability == "unfavorable") or ((len(risk_indicators) >= 2) or (risk_score >= 50)):
            return "medium"

        else:
            return "low"

    
    def _map_risk_score_to_level(self, risk_score: float) -> str:
        """
        Map numeric risk score to risk level string
        """
        if (risk_score >= 70):
            return "critical"

        elif (risk_score >= 50):
            return "high" 

        elif (risk_score >= 30):
            return "medium"
            
        else:
            return "low"
    

    def _fallback_interpretation(self, clause: ExtractedClause) -> ClauseInterpretation:
        """
        Fallback rule-based interpretation with risk context
        """
        category_summaries = {"compensation"          : "This clause defines payment terms, including salary, bonuses, and benefits.",
                              "termination"           : "This clause specifies conditions for ending the agreement, including notice periods and grounds for termination.",
                              "non_compete"           : "This clause restricts future employment opportunities with competitors.",
                              "confidentiality"       : "This clause requires protection of sensitive business information.",
                              "indemnification"       : "This clause defines financial responsibility for claims or losses.",
                              "intellectual_property" : "This clause determines ownership rights for work created.",
                              "liability"             : "This clause limits financial exposure for damages or breaches.",
                              "warranty"              : "This clause contains promises about quality or performance.",
                              "dispute_resolution"    : "This clause outlines processes for resolving disagreements.",
                             }
        
        summary            = category_summaries.get(clause.category, f"This {clause.category} clause defines specific rights and obligations.")
        
        key_points         = [f"Classified as {clause.category} clause",
                              f"Reference: {clause.reference}",
                              f"Extraction confidence: {clause.confidence:.2f}"
                             ]
        
        if clause.risk_indicators:
            key_points.append(f"Risk indicators: {', '.join(clause.risk_indicators[:3])}")
        
        potential_risks = clause.risk_indicators[:4] if clause.risk_indicators else ["Standard clause - review recommended"]
        
        # Favorability based on risk indicators and score
        risk_score = getattr(clause, 'risk_score', 0)
        
        if (len(clause.risk_indicators) >= 3) or (risk_score >= 70):
            favorability = "unfavorable"

        elif (len(clause.risk_indicators) >= 1) or (risk_score >= 40):
            favorability = "neutral"

        else:
            favorability = "favorable"
        
        negotiation_priority   = self._calculate_negotiation_priority(favorability    = favorability, 
                                                                      risk_indicators = clause.risk_indicators, 
                                                                      risk_score      = risk_score,
                                                                     )
        
        suggested_improvements = ["Review with legal counsel",
                                  "Compare with industry standards",
                                  "Consider impact on business operations"
                                 ]
        
        return ClauseInterpretation(clause_reference       = clause.reference,
                                    original_text          = clause.text[:500] + "..." if len(clause.text) > 500 else clause.text,
                                    plain_english_summary  = summary,
                                    key_points             = key_points,
                                    potential_risks        = potential_risks,
                                    favorability           = favorability,
                                    confidence_score       = 0.50,  # Medium confidence for fallback
                                    risk_score             = risk_score,
                                    negotiation_priority   = negotiation_priority,
                                    suggested_improvements = suggested_improvements,
                                   )
    

    def _generate_overall_risk_explanation(self, overall_risk_score: int, contract_type: ContractType, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection], 
                                           provider: LLMProvider) -> str:
        """
        Generate concise overall risk explanation
        """
        # Handle both object and dictionary formats for unfavorable_terms
        critical_terms       = list()
        high_terms           = list()
        issues_summary       = list()
        critical_protections = list()
        
        for term in unfavorable_terms:
            severity = ""
            
            if isinstance(term, UnfavorableTerm):
                severity = term.severity
            
            elif isinstance(term, dict):
                severity = term.get('severity', '')
            
            else:
                severity = getattr(term, 'severity', '')
                
            if (severity == "critical"):
                critical_terms.append(term)
            
            elif (severity == "high"):
                high_terms.append(term)
        
        # Handle both object and dictionary formats for missing_protections
        for protection in missing_protections:
            importance = ""

            if isinstance(protection, MissingProtection):
                importance = protection.importance
            
            elif isinstance(protection, dict):
                importance = protection.get('importance', '')
            
            else:
                importance = getattr(protection, 'importance', '')
                
            if (importance == "critical"):
                critical_protections.append(protection)
        
        # Create issues summary
        if critical_terms:
            issues_summary.append(f"{len(critical_terms)} CRITICAL unfavorable terms")
        
        if high_terms:
            issues_summary.append(f"{len(high_terms)} HIGH-risk unfavorable terms")
        
        if critical_protections:
            issues_summary.append(f"{len(critical_protections)} CRITICAL missing protections")
        
        if not issues_summary:
            issues_summary = ["Multiple concerning provisions identified"]
        
        prompt = f"""
                   Risk Level: {overall_risk_score}/100 for {contract_type.value} contract

                   Top Issues:
                   {chr(10).join(issues_summary)}

                   Write ONE sentence (max 25 words) explaining what this risk score means for someone signing this contract.

                   Example: "This contract creates severe financial and legal exposure through unlimited liability and one-sided termination rights."

                   Your turn:
                """
                                        
        try:
            response = self.llm_manager.complete(prompt      = prompt,
                                                 provider    = provider,
                                                 temperature = 0.2,
                                                 max_tokens  = 100,
                                                ) 
            
            explanation = response.text.strip() if response.success else self._fallback_risk_explanation(overall_risk_score)
            
            # Ensure single sentence
            sentences = explanation.split('.')
            return sentences[0].strip() + '.' if sentences else explanation
            
        except Exception as e:
            log_error(e, context={"operation": "generate_overall_risk_explanation"})
            return self._fallback_risk_explanation(overall_risk_score)
    

    def _fallback_risk_explanation(self, risk_score: int) -> str:
        """
        Fallback risk explanation
        """
        if (risk_score >= 80):
            return "This contract presents very high risk with multiple critical issues that require immediate attention and significant negotiation."

        elif (risk_score >= 60):
            return "This contract has substantial risk factors that need careful review and important modifications before signing."

        elif (risk_score >= 40):
            return "This contract has moderate risk with some areas that should be reviewed and potentially improved."

        else:
            return "This contract appears to have reasonable risk levels, but professional review is still recommended."
    

    def _extract_key_concerns(self, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection], clause_interpretations: List[ClauseInterpretation]) -> List[str]:
        """
        Extract key concerns from all analysis results
        """
        concerns       = list()
        
        # From unfavorable terms
        critical_terms = list()

        for term in unfavorable_terms:
            if isinstance(term, UnfavorableTerm):
                if (term.severity == "critical"):
                    critical_terms.append(term)
            
            elif isinstance(term, dict):
                if (term.get("severity") == "critical"):
                    critical_terms.append(term)
        
        # Top 10 critical terms
        for term in critical_terms[:10]:  
            term_name        = ""
            term_explanation = ""
            
            if isinstance(term, UnfavorableTerm):
                term_name        = term.term
                term_explanation = term.explanation
            
            elif isinstance(term, dict):
                term_name        = term.get('term', 'Unfavorable term')
                term_explanation = term.get('explanation', 'Standard risk identified')

            concerns.append(f"Critical: {term_name} - {term_explanation}")
        
        # From missing protections
        critical_protections = list()

        for protection in missing_protections:
            if isinstance(protection, MissingProtection):
                if (protection.importance == "critical"):
                    critical_protections.append(protection)
            
            elif isinstance(protection, dict):
                if (protection.get("importance") == "critical"):
                    critical_protections.append(protection)
        
        # Top 10 critical protections
        for protection in critical_protections[:10]:  
            protection_name = ""
            
            if isinstance(protection, MissingProtection):
                protection_name = protection.protection
            
            elif isinstance(protection, dict):
                protection_name = protection.get('protection', 'Critical protection')

            concerns.append(f"Missing: {protection_name}")
        
        # From clause interpretations
        high_priority_clauses = [c for c in clause_interpretations if (c.negotiation_priority == "high")]
        
        # Top 10 high priority clauses
        for clause in high_priority_clauses[:10]:  
            concerns.append(f"High priority: {clause.clause_reference} - {clause.plain_english_summary}")
        
        # Return top 20 concerns
        return concerns[:20]

    
    def _generate_negotiation_strategy(self, contract_type: ContractType, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection],
                                       overall_risk_score: int, provider: LLMProvider) -> str:
        """
        Generate negotiation strategy using LLM
        """
        prompt = f"""
                     As a negotiation expert, provide strategic advice for contract negotiations.

                     CONTRACT TYPE: {contract_type.value}
                     RISK LEVEL: {overall_risk_score}/100
                     KEY ISSUES: {len(unfavorable_terms)} unfavorable terms, {len(missing_protections)} missing protections

                     Provide 3-4 bullet points of negotiation strategy focusing on the most critical issues. Be practical and actionable.

                     Negotiation Strategy:
                  """
        
        try:
            response = self.llm_manager.complete(prompt      = prompt,
                                                 provider    = provider,
                                                 temperature = 0.3,
                                                 max_tokens  = 400,
                                                )
            
            return response.text.strip() if response.success else "Focus negotiation on the highest risk terms and missing critical protections identified in the analysis."
            
        except Exception as e:
            log_error(e, context = {"operation": "generate_negotiation_strategy"})
            return "Prioritize addressing critical risk terms and essential missing protections during negotiations."
    

    def _generate_market_comparison(self, contract_type: ContractType, overall_risk_score: int, provider: LLMProvider) -> str:
        """
        Generate market comparison context
        """
        prompt = f"""
                     Provide market context for this contract type.

                     CONTRACT TYPE: {contract_type.value}
                     RISK SCORE: {overall_risk_score}/100

                     How does this risk level compare to typical market standards for this type of contract? Provide 1-2 sentences of context.

                     Market Comparison:
                  """
                            
        try:
            response = self.llm_manager.complete(prompt      = prompt,
                                                 provider    = provider,
                                                 temperature = 0.2,
                                                 max_tokens  = 200,
                                                )
            
            return response.text.strip() if response.success else "Compare with industry standards for similar contracts."
            
        except Exception as e:
            log_error(e, context = {"operation": "generate_market_comparison"})
            return "Review against industry benchmarks for this contract type."


    def interpret_specific_clause(self, clause_text: str, clause_reference: str = "Unknown", category: str = "general", provider: Optional[LLMProvider] = None) -> ClauseInterpretation:
        """
        Interpret a specific clause text directly
        """
        temp_clause = ExtractedClause(text              = clause_text,
                                      reference         = clause_reference,
                                      category          = category,
                                      confidence        = 1.0,
                                      start_pos         = 0,
                                      end_pos           = len(clause_text),
                                      extraction_method = "manual",
                                      risk_indicators   = [],
                                      legal_bert_score  = 0.0,
                                     )
        
        return self._interpret_single_clause(temp_clause, provider or self.default_provider)
    
    
    def batch_interpret(self, clauses: List[ExtractedClause], provider: Optional[LLMProvider] = None) -> List[ClauseInterpretation]:
        """
        Batch interpretation with progress tracking
        """
        return self.interpret_clauses(clauses     = clauses,
                                      max_clauses = len(clauses),
                                      provider    = provider,
                                     )
    

    def get_unfavorable_interpretations(self, interpretations: List[ClauseInterpretation]) -> List[ClauseInterpretation]:
        """
        Filter to only unfavorable clause interpretations
        """
        unfavorable = [i for i in interpretations if (i.favorability == "unfavorable")]
        log_info(f"Found {len(unfavorable)} unfavorable interpretations")
        
        return unfavorable

    
    def get_high_risk_interpretations(self, interpretations: List[ClauseInterpretation], min_risk_count: int = 2) -> List[ClauseInterpretation]:
        """
        Filter to interpretations with multiple risks
        """
        high_risk = [i for i in interpretations if (len(i.potential_risks) >= min_risk_count)]
        log_info(f"Found {len(high_risk)} high-risk interpretations")

        return high_risk