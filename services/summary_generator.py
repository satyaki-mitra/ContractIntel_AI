# DEPENDENCIES
import sys
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.risk_analyzer import RiskScore
from services.data_models import SummaryContext
from utils.logger import ContractAnalyzerLogger 
from model_manager.llm_manager import LLMManager
from model_manager.llm_manager import LLMProvider
from services.data_models import ContractCategory
from services.data_models import RiskInterpretation
from services.data_models import NegotiationPlaybook


class SummaryGenerator:
    """
    LLM-powered executive summary generator for contract analysis : Generates professional, detailed executive summaries using ALL pipeline outputs
    """
    def __init__(self, llm_manager: Optional[LLMManager] = None, default_provider: Optional[LLMProvider] = None):
        """
        Initialize the summary generator
        
        Arguments:
        ----------
            llm_manager       { LLMManager }  : LLM manager instance (if None, creates one with default settings)

            default_provider  { LLMProvider } : Default LLM provider to use if creating new LLMManager
        """
        # Create LLMManager with the specified provider (or use default from settings)
        if llm_manager is None:
            self.llm_manager = LLMManager(default_provider = default_provider)
        
        else:
            self.llm_manager = llm_manager

        self.logger      = ContractAnalyzerLogger.get_logger() 
        
        self.logger.info("Summary generator initialized")


    # Main entry point with full pipeline integration
    def generate_executive_summary(self, contract_text: str, classification: ContractCategory, risk_analysis: RiskScore, risk_interpretation: RiskInterpretation,
                                   negotiation_playbook: NegotiationPlaybook, unfavorable_terms: List, missing_protections: List, clauses: List,
                                   provider: Optional[LLMProvider] = None) -> str:
        """
        Generate executive summary using all the pipeline outputs
        
        Arguments:
        ----------
            contract_text               { str }          : Original contract text (for context)
            
            classification       { ContractCategory }    : Contract classification results
            
            risk_analysis            { RiskScore }       : Complete risk analysis
            
            risk_interpretation  { RiskInterpretation }  : LLM-enhanced risk explanations
            
            negotiation_playbook { NegotiationPlaybook } : Comprehensive negotiation strategy
            
            unfavorable_terms            { List }        : Detected unfavorable terms
            
            missing_protections          { List }        : Missing protections
            
            clauses                      { List }        : Extracted clauses

            provider                 { LLMProvide }      : Optional LLM provider override
            
        Returns:
        --------
                             { str }                     : Generated executive summary string
        """
        try:
            # Prepare context with all pipeline data
            context = self._prepare_summary_context(contract_text        = contract_text,
                                                    classification       = classification,
                                                    risk_analysis        = risk_analysis,
                                                    risk_interpretation  = risk_interpretation,
                                                    negotiation_playbook = negotiation_playbook,
                                                    unfavorable_terms    = unfavorable_terms,
                                                    missing_protections  = missing_protections,
                                                    clauses              = clauses,
                                                   )
            
            # Generate summary using LLM
            summary = self._generate_summary(context  = context, 
                                             provider = provider,
                                            )
            
            self.logger.info(f"Executive summary generated - Risk: {context.risk_score}/100 ({context.risk_level})") 
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive summary: {repr(e)}") 
            
            # Fallback with available data
            return self._generate_fallback_summary(contract_text       = contract_text,
                                                   classification      = classification,
                                                   risk_analysis       = risk_analysis,
                                                   unfavorable_terms   = unfavorable_terms,
                                                   missing_protections = missing_protections,
                                                  )
    

    def _prepare_summary_context(self, contract_text: str, classification: ContractCategory, risk_analysis: RiskScore, risk_interpretation: RiskInterpretation,
                                 negotiation_playbook: NegotiationPlaybook, unfavorable_terms: List[Dict], missing_protections: List[Dict], clauses: List) -> SummaryContext:
        """
        Prepare summary context with all pipeline data
        """
        # Handle null negotiation_playbook
        walk_away_count = 0

        if negotiation_playbook and hasattr(negotiation_playbook, 'walk_away_items'):
            walk_away_count = len(negotiation_playbook.walk_away_items)
            
        # Extract contract text
        contract_preview = contract_text 
        
        # Extract key findings from all sources
        key_findings     = self._extract_findings(risk_analysis        = risk_analysis,
                                                  risk_interpretation  = risk_interpretation,
                                                  negotiation_playbook = negotiation_playbook,
                                                  unfavorable_terms    = unfavorable_terms,
                                                  missing_protections  = missing_protections,
                                                  clauses              = clauses,
                                                 )
                                                            
        # Prepare metadata
        metadata         = {"contract_length"  : len(contract_text),
                            "clauses_analyzed" : len(clauses),
                            "critical_issues"  : len([t for t in unfavorable_terms if (self._get_severity(t) == "critical")]),
                            "walk_away_items"  : walk_away_count,
                           }
         
        return SummaryContext(contract_type         = classification.category,
                              risk_score            = risk_analysis.overall_score,
                              risk_level            = risk_analysis.risk_level,
                              category_scores       = risk_analysis.category_scores,
                              unfavorable_terms     = unfavorable_terms,
                              missing_protections   = missing_protections,
                              clauses               = clauses,
                              key_findings          = key_findings,
                              risk_interpretation   = risk_interpretation,
                              negotiation_playbook  = negotiation_playbook,
                              contract_text_preview = contract_preview,
                              contract_metadata     = metadata,
                             )

    
    def _extract_findings(self, risk_analysis: RiskScore, risk_interpretation: RiskInterpretation, negotiation_playbook: NegotiationPlaybook,
                          unfavorable_terms: List[Dict], missing_protections: List[Dict], clauses: List) -> List[str]:
        """
        Extract findings from all analysis components
        """
        findings = list()
        
        # Overall risk context
        if (risk_analysis.overall_score >= 80):
            findings.append("CRITICAL RISK LEVEL: Contract presents unacceptable risk requiring immediate attention")

        elif (risk_analysis.overall_score >= 60):
            findings.append("HIGH RISK LEVEL: Significant concerns requiring substantial negotiation")
        
        # Critical unfavorable terms
        critical_terms = [t for t in unfavorable_terms if (self._get_severity(t) == "critical")]
        
        if critical_terms:
            findings.append(f"{len(critical_terms)} CRITICAL unfavorable terms identified")
            for term in critical_terms[:2]:
                term_name = self._get_term_name(term = term)
                
                findings.append(f"Critical: {term_name}")
        
        # Critical missing protections
        critical_protections = [p for p in missing_protections if (self._get_importance(p) == "critical")]
        
        if critical_protections:
            findings.append(f"{len(critical_protections)} CRITICAL protections missing")
            for prot in critical_protections[:2]:
                prot_name = self._get_protection_name(protection = prot)
                
                findings.append(f"Missing: {prot_name}")
        
        # High-risk categories
        high_risk_categories = [cat for cat, score in risk_analysis.category_scores.items() if (score >= 70)]
        if high_risk_categories:
            findings.append(f"High-risk categories: {', '.join(high_risk_categories)}")
        
        # Walk-away items from negotiation playbook
        if negotiation_playbook and negotiation_playbook.walk_away_items:
            findings.append(f"{len(negotiation_playbook.walk_away_items)} potential deal-breakers identified")
        
        # Key concerns from risk interpretation
        if risk_interpretation and risk_interpretation.key_concerns:
            top_concerns = risk_interpretation.key_concerns[:2]
            for concern in top_concerns:
                findings.append(f"Key concern: {concern}")
        
        return findings  
    

    def _generate_summary(self, context: SummaryContext, provider: Optional[LLMProvider] = None) -> str:
        """
        Generate enhanced summary using comprehensive context
        """
        prompt        = self._build_summary_prompt(context)
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.llm_manager.complete(prompt        = prompt,
                                                 system_prompt = system_prompt,
                                                 provider      = provider,
                                                 temperature   = 0.3,
                                                 max_tokens    = 500, 
                                                 json_mode     = False,
                                                )
              
            if response.success and response.text.strip():
                return self._clean_summary_response(text = response.text)
            
            else:
                raise ValueError(f"LLM generation failed: {response.error_message}")
                
        except Exception as e:
            self.logger.error(f"Enhanced LLM summary generation failed: {repr(e)}")
            # Fallback to basic summary
            return self._generate_fallback_summary_from_context(context = context)
    

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for executive summary generation
        """
        system_prompt =  """
                            You are a senior contract risk analyst. Generate CONCISE executive summaries.

                            CRITICAL REQUIREMENTS:
                            1. Maximum 120 words (strict limit)
                            2. Must mention SPECIFIC clause numbers (e.g., Clause 8.2, Clause 9.5)
                            3. Direct, urgent tone - no hedging or academic language
                            4. Focus ONLY on top 3 critical risks

                            STRUCTURE (3-4 sentences total):
                            Sentence 1: Overall risk assessment with contract type
                            Sentence 2-3: Top 2-3 critical risks with SPECIFIC clause references
                            Sentence 4: Brief actionable conclusion

                            TONE EXAMPLES:
                            ✅ GOOD: "This employment agreement is heavily skewed in favor of the Employer. Clause 8.2 fails to define post-probation salary. Clause 11.2 allows illegal wage forfeiture."
                            ❌ BAD: "The comprehensive analysis indicates that there are several concerns that require attention. It is essential to carefully review..."

                            FORBIDDEN PHRASES:
                            - "comprehensive analysis"
                            - "it is essential to"
                            - "requires attention"
                            - "should be reviewed"
                            - "it is recommended"

                            OUTPUT: Pure paragraph text only. No formatting, no bullets, no headers.
                         """
        
        return system_prompt


    def _build_summary_prompt(self, context: SummaryContext) -> str:
        """
        Build prompt for executive summary generation
        """
        # Extract top critical issues only
        critical_terms       = [t for t in context.unfavorable_terms if self._get_severity(t) == "critical"]
        
        critical_protections = [p for p in context.missing_protections if self._get_importance(p) == "critical"]
        
        # Build concise context
        critical_issues_text = ""
        
        if critical_terms:
            critical_issues_text += "CRITICAL UNFAVORABLE TERMS:\n"
            
            for term in critical_terms:
                clause_reference      = self._get_clause_reference(term = term)
                term_name             = self._get_term_name(term = term)
                critical_issues_text += f"- {clause_reference}: {term_name}\n"
        
        if critical_protections:
            critical_issues_text += "\nCRITICAL MISSING PROTECTIONS:\n"
            
            for protection in critical_protections:
                protection_name       = self._get_protection_name(protection = protection)
                critical_issues_text += f"- {protection_name}\n"
        
        # Determine risk tone
        if (context.risk_score >= 80):
            risk_tone = "heavily skewed/very high risk/presents unacceptable risk"
        
        elif (context.risk_score >= 60):
            risk_tone = "significantly unfavorable/high risk/substantial concerns"

        elif (context.risk_score >= 40):
            risk_tone = "moderately concerning/notable risk/requires negotiation"

        else:
            risk_tone = "generally reasonable/manageable risk/standard concerns"
        
        summary_prompt = f"""
                             CONTRACT ANALYSIS DATA:
                            
                             - Type: {context.contract_type.replace('_', ' ').title()}
                             - Risk Score: {context.risk_score}/100
                             - Risk Level: {context.risk_level}
                             - Appropriate Tone: {risk_tone}

                             {critical_issues_text}

                             TASK:
                             Write a 100-120 word executive summary following this EXACT structure:

                             1. First sentence: "This [contract type] [risk assessment with tone matching score]"
                             2. Second sentence: State top critical risk with SPECIFIC clause number
                             3. Third sentence: State second critical risk with SPECIFIC clause number
                             4. Fourth sentence: Brief conclusion about action needed

                             EXAMPLE (for 85/100 risk employment contract):
                             "This employment agreement is heavily skewed in favor of the Employer, presenting a very high risk to the Employee. Key concerns include Clause 9.5's extremely broad 24-month non-compete against the entire industry, and Clause 11.2's punitive penalty allowing forfeiture of earned wages. The termination clauses in Clause 17 are highly asymmetrical, giving the employer unilateral power. Significant negotiation is required before signing."

                             YOUR TURN - Generate summary for THIS contract:
                          """
                        
        return summary_prompt
    

    def _clean_summary_response(self, text: str) -> str:
        """
        Clean and format the LLM response
        """
        # Remove any markdown formatting
        text          = text.replace('**', '').replace('*', '').replace('#', '')
        
        # Remove common LLM artifacts and empty lines
        lines         = text.split('\n')
        cleaned_lines = list()
        
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('executive summary', 'summary:', 'here is', 'based on', 'certainly')):
                cleaned_lines.append(line)
        
        # Join into coherent paragraph
        summary = ' '.join(cleaned_lines)
        
        # Ensure proper sentence structure
        if summary:
            if not summary[0].isupper():
                summary = summary[0].upper() + summary[1:]
            
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
        
        return summary
    

    def _generate_fallback_summary(self, contract_text: str, classification: ContractCategory, risk_analysis: RiskScore, unfavorable_terms: List[Dict], missing_protections: List[Dict]) -> str:
        """
        Generate enhanced fallback summary
        """
        contract_type_display = classification.category.replace('_', ' ').title()
        
        # Count critical items
        critical_terms        = len([t for t in unfavorable_terms if (self._get_severity(t) == "critical")])
        critical_protections  = len([p for p in missing_protections if (self._get_importance(p) == "critical")])
        
        # Risk assessment
        if (risk_analysis.overall_score >= 80):
            risk_assessment = f"This {contract_type_display} presents a CRITICAL level of risk"
            action          = "requires immediate executive attention and significant revision before consideration"
        
        elif (risk_analysis.overall_score >= 60):
            risk_assessment = f"This {contract_type_display} presents a HIGH level of risk" 
            action          = "requires careful legal review and substantial negotiation to mitigate key concerns"
        
        elif (risk_analysis.overall_score >= 40):
            risk_assessment = f"This {contract_type_display} presents a MODERATE level of risk"
            action          = "requires professional review and selective negotiation on specific provisions"
        
        else:
            risk_assessment = f"This {contract_type_display} presents a LOW level of risk"
            action = "appears generally reasonable but should undergo standard legal review"
        
        summary  = f"{risk_assessment} with an overall risk score of {risk_analysis.overall_score}/100. "
        summary += f"The agreement {action}. "
        
        # Add critical items context
        if (critical_terms > 0):
            summary += f"Analysis identified {critical_terms} critical unfavorable terms "

            if critical_protections > 0:
                summary += f"and {critical_protections} critical missing protections. "

            else:
                summary += f"and {len(missing_protections)} missing standard protections. "

        else:
            summary += f"Review identified {len(unfavorable_terms)} areas for improvement. "
        
        # Add high-risk categories context
        high_risk_categories = [cat for cat, score in risk_analysis.category_scores.items() if (score >= 60)]
        
        if high_risk_categories:
            category_names = [cat.replace('_', ' ').title() for cat in high_risk_categories[:2]]
            summary       += f"Particular attention should be given to {', '.join(category_names)} provisions. "
        
        summary += "Proceed with the detailed negotiation strategy and risk mitigation recommendations provided in the full analysis."
        
        return summary

    
    def _generate_fallback_summary_from_context(self, context: SummaryContext) -> str:
        """
        Generate fallback summary from context object
        """
        # Access attributes safely, providing defaults if needed by the fallback logic
        text_preview  = context.contract_text_preview if context.contract_text_preview is not None else ""
        missing_prots = context.missing_protections if context.missing_protections is not None else []
        unfav_terms   = context.unfavorable_terms if context.unfavorable_terms is not None else []

        return self._generate_fallback_summary(contract_text       = text_preview,
                                               classification      = type('MockClassification', (), {'category': context.contract_type})(),
                                               risk_analysis       = type('MockRiskAnalysis', (), {'overall_score': context.risk_score, 'risk_level': context.risk_level, 'category_scores': context.category_scores or {}})(),
                                               unfavorable_terms   = unfav_terms,
                                               missing_protections = missing_prots,
                                              )
    

    def _get_severity(self, term) -> str:
        """
        Safely get severity from term object or dict
        """
        try:
            if (hasattr(term, 'severity')):
                return term.severity
            
            else:
                return term.get('severity', 'unknown')
        
        except (AttributeError, KeyError):
            return 'unknown'
    

    def _get_importance(self, protection) -> str:
        """
        Safely get importance from protection object or dict
        """
        try:
            if hasattr(protection, 'importance'):
                return protection.importance

            else:
                return protection.get('importance', 'unknown')
        
        except (AttributeError, KeyError):
            return 'unknown'
    

    def _get_term_name(self, term) -> str:
        """
        Safely get term name
        """
        try:
            if hasattr(term, 'term'):
                return term.term

            else:
                return term.get('term', 'Unknown Term')

        except (AttributeError, KeyError):
            return 'Unknown Term'
    

    def _get_protection_name(self, protection) -> str:
        """
        Safely get protection name
        """
        try:
            if hasattr(protection, 'protection'):
                return protection.protection
            
            else:
                return protection.get('protection', 'Unknown Protection')
        
        except (AttributeError, KeyError):
            return 'Unknown Protection'
    

    def _get_explanation(self, item) -> str:
        """
        Safely get explanation
        """
        try:
            if hasattr(item, 'explanation'):
                return item.explanation
            
            else:
                return item.get('explanation', 'No explanation available')
        
        except (AttributeError, KeyError):
            return 'No explanation available'


    def _get_clause_reference(self, term) -> str:
        """
        Safely get clause reference from term
        """
        try:
            if hasattr(term, 'clause_reference'):
                ref = term.clause_reference
                return ref if ref and ref != 'None' else 'Multiple clauses'

            else:
                ref = term.get('clause_reference', '')
                return ref if ref and ref != 'None' else 'Multiple clauses'

        except (AttributeError, KeyError):
            return 'Unknown clause'