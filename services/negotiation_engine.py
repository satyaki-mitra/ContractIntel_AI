# DEPENDENCIES
import re
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
from services.risk_analyzer import RiskScore
from utils.logger import ContractAnalyzerLogger
from model_manager.llm_manager import LLMManager
from services.data_models import UnfavorableTerm
from model_manager.llm_manager import LLMProvider
from services.data_models import NegotiationPoint
from services.data_models import NegotiationTactic
from services.data_models import MissingProtection
from services.data_models import NegotiationPlaybook
from services.clause_extractor import ExtractedClause
from services.llm_interpreter import RiskInterpretation
from services.llm_interpreter import ClauseInterpretation
from services.protection_checker import MissingProtection


class NegotiationEngine:
    """
    Generate intelligent negotiation strategy with LLM enhancement integrated with full analysis pipeline and RiskRules framework
    """
    def __init__(self, llm_manager: LLMManager, default_provider: LLMProvider = LLMProvider.OLLAMA):
        """
        Initialize negotiation engine
        Arguments:
        ----------
            llm_manager      { LLMManager }  : LLMManager instance
            
            default_provider { LLMProvider } : Default LLM provider
        """
        self.llm_manager      = llm_manager
        self.default_provider = default_provider
        self.risk_rules       = RiskRules()
        self.logger           = ContractAnalyzerLogger.get_logger()

        log_info("NegotiationEngine initialized", default_provider = default_provider.value)


    @ContractAnalyzerLogger.log_execution_time("generate_comprehensive_playbook")
    def generate_comprehensive_playbook(self, risk_analysis: RiskScore, risk_interpretation: RiskInterpretation, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection],
                                        clauses: List[ExtractedClause], contract_type: ContractType, max_points: int = 10, provider: Optional[LLMProvider] = None) -> NegotiationPlaybook:
        """
        Generate comprehensive negotiation playbook using all analysis results

        Arguments:
        ----------
            risk_analysis               : Complete risk analysis
            
            risk_interpretation         : LLM-enhanced risk explanations
            
            unfavorable_terms           : Detected unfavorable terms
            
            missing_protections         : Missing protections
            
            clauses                     : Extracted clauses with risk scores (should be risk-category based)
            
            contract_type               : Contract type for strategy
            
            max_points                  : Maximum negotiation points
            
            provider                    : LLM provider
        
        Returns:
        --------
            { NegotiationPlaybook }     : Comprehensive NegotiationPlaybook
        """
        provider = provider or self.default_provider
        
        log_info("Starting comprehensive negotiation playbook generation", contract_type = contract_type.value, overall_risk = risk_analysis.overall_score, max_points = max_points)
        
        # Generate prioritized negotiation points
        negotiation_points   = self.generate_negotiation_points(risk_analysis       = risk_analysis,
                                                                unfavorable_terms   = unfavorable_terms,
                                                                missing_protections = missing_protections,
                                                                clauses             = clauses,
                                                                max_points          = max_points,
                                                                provider            = provider,
                                                               )
        
        # Generate overall strategy using LLM
        overall_strategy     = self._generate_overall_strategy(risk_analysis       = risk_analysis,
                                                               risk_interpretation = risk_interpretation,
                                                               contract_type       = contract_type,
                                                               provider            = provider,
                                                              )
        
        # Identify walk-away items
        walk_away_items      = self._identify_walk_away_items(negotiation_points = negotiation_points,
                                                              risk_analysis      = risk_analysis,
                                                             )
        
        # Identify concession items
        concession_items     = self._identify_concession_items(negotiation_points = negotiation_points,
                                                               risk_analysis      = risk_analysis,
                                                              )
        
        # Generate timing guidance
        timing_guidance      = self._generate_timing_guidance(negotiation_points = negotiation_points,
                                                              contract_type      = contract_type,
                                                              provider           = provider,
                                                             )
        
        # Generate risk mitigation plan
        risk_mitigation_plan = self._generate_risk_mitigation_plan(risk_analysis      = risk_analysis,
                                                                   negotiation_points = negotiation_points,
                                                                   provider           = provider,
                                                                  )
        
        playbook             = NegotiationPlaybook(overall_strategy     = overall_strategy,
                                                   critical_points      = negotiation_points,
                                                   walk_away_items      = walk_away_items,
                                                   concession_items     = concession_items,
                                                   timing_guidance      = timing_guidance,
                                                   risk_mitigation_plan = risk_mitigation_plan,
                                                  )
        log_info("Comprehensive negotiation playbook generated", critical_points = len(negotiation_points), walk_away_items = len(walk_away_items))
        return playbook


    @ContractAnalyzerLogger.log_execution_time("generate_negotiation_points")
    def generate_negotiation_points(self, risk_analysis: RiskScore, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection],
                                    clauses: List[ExtractedClause], max_points: int = 10, provider: Optional[LLMProvider] = None) -> List[NegotiationPoint]:
        """
        Generate prioritized negotiation strategy

        Arguments:
        ----------
            risk_analysis       { RiskScore }   : Risk analysis results
            
            unfavorable_terms     { list }      : Detected unfavorable terms
            
            missing_protections   { list }      : Missing protections
            
            clauses               { list }      : Extracted clauses (ideally with risk categories)
            
            max_points            { int }       : Maximum negotiation points to generate
            
            provider           { LLMProvider }  : LLM provider
        
        Returns:
        --------
                        { list }                : List of NegotiationPoint objects sorted by priority
        """
        provider                               = provider or self.default_provider
        
        # Convert dictionaries to objects if needed
        unfavorable_terms, missing_protections = self._ensure_objects(unfavorable_terms, missing_protections)
        
        log_info("Starting negotiation points generation", max_points = max_points, unfavorable_terms = len(unfavorable_terms), missing_protections = len(missing_protections))
        
        negotiation_points                     = list()
        
        # Critical unfavorable terms (walk-away level)
        critical_terms                         = [t for t in unfavorable_terms if (t.severity == "critical")]
        
        # Top-10 critical terms
        for term in critical_terms[:10]:  
            point = self._create_enhanced_point_from_term(term, clauses, priority = 1)
            if point:
                negotiation_points.append(point)
        
        # Critical missing protections
        critical_protections = [p for p in missing_protections if (p.importance == "critical")]
        
        for protection in critical_protections[:10]:
            point = self._create_enhanced_point_from_protection(protection, priority = 2)
            negotiation_points.append(point)
        
        # High unfavorable terms
        high_terms = [t for t in unfavorable_terms if (t.severity == "high")]
        for term in high_terms[:10]:
            point = self._create_enhanced_point_from_term(term, clauses, priority = 3)
            if point:
                negotiation_points.append(point)
        
        # High-risk categories from risk analysis
        high_risk_categories = self._get_high_risk_categories(risk_analysis)
        for category in high_risk_categories[:10]:
            point = self._create_category_strategy_point(category, risk_analysis, clauses, priority = 4)
            if point:
                negotiation_points.append(point)
        
        # Medium unfavorable terms and missing protections
        medium_terms = [t for t in unfavorable_terms if (t.severity == "medium")]
        for term in medium_terms[:10]:
            point = self._create_enhanced_point_from_term(term, clauses, priority=5)
            if point:
                negotiation_points.append(point)
        
        medium_protections = [p for p in missing_protections if (p.importance == "medium")]
        for protection in medium_protections[:10]:
            point = self._create_enhanced_point_from_protection(protection, priority = 5)
            negotiation_points.append(point)
        
        # Enhance with LLM for sophisticated language and strategy
        enhanced_points = self._enhance_with_llm_strategy(negotiation_points[:max_points], 
                                                          risk_analysis,
                                                          provider,
                                                         )
        log_info(f"Negotiation points generation complete", total_points = len(enhanced_points))
        
        return enhanced_points[:max_points]


    def _create_enhanced_point_from_term(self, term: UnfavorableTerm, clauses: List[ExtractedClause], priority: int) -> Optional[NegotiationPoint]:
        """
        Create enhanced negotiation point from unfavorable term from the clauses that are extracted by from RiskClauseExtractor having risk categories
        """
        # Find clause by reference
        clause = next((c for c in clauses if (c.reference == term.clause_reference)), None)
        
        # If not found by reference, try finding by matching risk category (if term.category is a risk category)
        if not clause:
            clause = next((c for c in clauses if (c.category == term.category)), None) # term.category should be a risk category from TermAnalyzer

        if not clause:
            log_info(f"Could not find clause for term: {term.term} in category: {term.category}", clause_reference=term.clause_reference)
            return None

        current               = clause.text
        
        # Determine negotiation tactic
        tactic                = self._determine_negotiation_tactic(term, clause)
        
        # Generate sophisticated proposed language
        proposed              = self._generate_enhanced_proposed_language(term, clause, tactic)
        
        # Calculate difficulty
        difficulty            = self._calculate_negotiation_difficulty(term, tactic)
        
        # Generate strategic context
        business_impact       = self._generate_business_impact(term, clause)
        counterparty_concerns = self._generate_counterparty_concerns(term, tactic)
        timing                = self._suggest_timing(priority, tactic)
        
        return NegotiationPoint(priority              = priority,
                                category              = term.category,
                                issue                 = term.term,
                                current_language      = current,
                                proposed_language     = proposed,
                                rationale             = term.explanation,
                                tactic                = tactic,
                                fallback_position     = self._generate_strategic_fallback(term, tactic),
                                estimated_difficulty  = difficulty,
                                legal_basis           = term.legal_basis,
                                business_impact       = business_impact,
                                counterparty_concerns = counterparty_concerns,
                                timing_suggestion     = timing,
                                bargaining_chips      = self._suggest_bargaining_chips(term, tactic),
                               )


    def _create_enhanced_point_from_protection(self, protection: MissingProtection, priority: int) -> NegotiationPoint:
        """
        Create enhanced negotiation point from missing protection
        """
        difficulty = "medium" if (protection.importance == "critical") else "easy"

        return NegotiationPoint(priority             = priority,
                                category             = protection.categories[0] if protection.categories else "general",
                                issue                = f"Add {protection.protection}",
                                current_language     = "[NOT PRESENT IN CONTRACT]",
                                proposed_language    = protection.suggested_language or protection.recommendation,
                                rationale            = protection.explanation,
                                tactic               = NegotiationTactic.ADDITION,
                                fallback_position    = self._generate_protection_fallback(protection),
                                estimated_difficulty = difficulty,
                                legal_basis          = protection.legal_basis,
                                business_impact      = f"Missing this protection creates {protection.risk_score}/100 risk exposure",
                                timing_suggestion    = "Early in negotiations - establishes baseline protections",
                                bargaining_chips     = ["Offer to review their standard protections in return"],
                               )


    def _create_category_strategy_point(self, category: str, risk_analysis: RiskScore, clauses: List[ExtractedClause], priority: int) -> Optional[NegotiationPoint]:
        """
        Create strategic negotiation point for high-risk category, where clauses are from RiskClauseExtractor and have risk categories
        """
        # Find clauses that belong to this *risk* category
        category_clauses = [c for c in clauses if c.category == category] # Direct match on risk category
        if not category_clauses:
            log_info(f"No clauses found for high-risk category: {category}", available_categories=[c.category for c in clauses])
            return None

        score            = risk_analysis.category_scores.get(category, 0)

        # Use high description as default for high-risk
        description      = self.risk_rules.CATEGORY_DESCRIPTIONS.get(category, {}).get("high", "") 
        
        return NegotiationPoint(priority             = priority,
                                category             = category,
                                issue                = f"Address {category.replace('_', ' ')} risks (score: {score}/100)",
                                current_language     = f"Multiple clauses in {category} category present elevated risk (e.g., {category_clauses[0].reference}).",
                                proposed_language    = f"Request balanced, market-standard terms for {category.replace('_', ' ')} provisions",
                                rationale            = description,
                                tactic               = NegotiationTactic.MODIFICATION,
                                estimated_difficulty = "medium",
                                business_impact      = f"High risk category affecting multiple contract areas",
                                timing_suggestion    = "Mid-negotiations after establishing rapport",
                               )


    def _determine_negotiation_tactic(self, term: UnfavorableTerm, clause: ExtractedClause) -> NegotiationTactic:
        """
        Determine the best negotiation tactic for this term
        """
        text_lower = clause.text.lower()
        if (("unlimited" in text_lower) or ("sole discretion" in text_lower)):
            return NegotiationTactic.LIMITATION
        
        elif (("indemnify" in text_lower) and ("mutual" not in text_lower)):
            return NegotiationTactic.MUTUALIZATION
        
        elif (any(word in text_lower for word in ["forfeit", "penalty", "liquidated damages"])):
            return NegotiationTactic.REMOVAL
        
        elif (("vague" in term.explanation.lower()) or ("ambiguous" in term.explanation.lower())):
            return NegotiationTactic.CLARIFICATION
        
        else:
            return NegotiationTactic.MODIFICATION


    def _generate_enhanced_proposed_language(self, term: UnfavorableTerm, clause: ExtractedClause, tactic: NegotiationTactic) -> str:
        """
        Generate sophisticated proposed language based on tactic
        """
        language_templates = {NegotiationTactic.REMOVAL       : "Remove the following language: '[EXTRACT PROBLEMATIC PHRASE]'",
                              NegotiationTactic.LIMITATION    : "Add limitation: 'Not to exceed [REASONABLE LIMIT]' or 'Subject to [REASONABLE STANDARD]'",
                              NegotiationTactic.MUTUALIZATION : "Make mutual: 'Each party shall [APPLY TO BOTH PARTIES]'",
                              NegotiationTactic.CLARIFICATION : "Clarify: 'For purposes of this section, [TERM] means [CLEAR DEFINITION]'",
                              NegotiationTactic.MODIFICATION  : "Modify to: '[BALANCED, MARKET-STANDARD LANGUAGE]'",
                             }

        base_template      = language_templates.get(tactic, term.suggested_fix or "[Request balanced language]")
        
        # Enhance with specific examples based on term type
        if ("non-compete" in term.term.lower()):
            return "Limit to: (a) 6-12 month duration, (b) direct competitors only, (c) reasonable geographic scope"
        
        elif ("liability" in term.term.lower()):
            return "Add: 'Total liability capped at the greater of $[AMOUNT] or fees paid in preceding 12 months'"
        
        elif ("termination" in term.term.lower()):
            return "Modify to provide mutual [30-60] day notice period and clear 'for cause' definition"
        
        return base_template


    def _calculate_negotiation_difficulty(self, term: UnfavorableTerm, tactic: NegotiationTactic) -> str:
        """
        Calculate negotiation difficulty
        """
        if ((term.severity == "critical") and (tactic == NegotiationTactic.REMOVAL)):
            return "hard"
        
        elif ((term.severity == "high") or (tactic == NegotiationTactic.MUTUALIZATION)):
            return "medium"
        
        else:
            return "easy"


    def _generate_business_impact(self, term: UnfavorableTerm, clause: ExtractedClause) -> str:
        """
        Generate business impact analysis
        """
        if (term.severity == "critical"):
            return "Could result in significant financial exposure or business restrictions"
        
        elif (term.severity == "high"):
            return "Creates substantial operational risk or compliance burden"
        
        else:
            return "Standard business risk that should be managed"


    def _generate_counterparty_concerns(self, term: UnfavorableTerm, tactic: NegotiationTactic) -> str:
        """
        Anticipate counterparty concerns
        """
        concerns = {NegotiationTactic.REMOVAL       : "They may view this as essential protection",
                    NegotiationTactic.LIMITATION    : "They may argue this undermines the provision's purpose", 
                    NegotiationTactic.MUTUALIZATION : "They may prefer one-sided advantage",
                    NegotiationTactic.CLARIFICATION : "They may prefer ambiguity for flexibility",
                   }
        
        return concerns.get(tactic, "Standard negotiation resistance expected")


    def _suggest_timing(self, priority: int, tactic: NegotiationTactic) -> str:
        """
        Suggest negotiation timing
        """
        if (priority <= 2):
            return "Address early - these are deal-breakers"
        
        elif (tactic == NegotiationTactic.ADDITION):
            return "Early in negotiations - establishes baseline"
        
        else:
            return "Mid-negotiations - after establishing key terms"


    def _suggest_bargaining_chips(self, term: UnfavorableTerm, tactic: NegotiationTactic) -> List[str]:
        """
        Suggest bargaining chips
        """
        chips = list()

        if (tactic == NegotiationTactic.REMOVAL):
            chips.append("Offer alternative protection that addresses their underlying concern")
        
        elif (tactic == NegotiationTactic.LIMITATION):
            chips.append("Accept their position with reasonable cap or standard")
        
        elif (tactic == NegotiationTactic.MUTUALIZATION):
            chips.append("Frame as fairness principle benefiting both parties")
        
        chips.append("Trade for lower priority item they care about")
        
        return chips


    def _generate_strategic_fallback(self, term: UnfavorableTerm, tactic: NegotiationTactic) -> str:
        """
        Generate strategic fallback position
        """
        if (term.severity == "critical"):
            return "If no compromise, seriously consider walking away - this creates unacceptable risk"
        
        elif (term.severity == "high"):
            return "If they refuse, document objection and consider risk mitigation strategies"
        
        else:
            return "If they won't budge, assess if other favorable terms compensate for this risk"


    def _ensure_objects(self, unfavorable_terms, missing_protections):
        """
        Convert dictionaries back to proper objects if needed
        """
        if (unfavorable_terms and isinstance(unfavorable_terms[0], dict)):
            
            unfavorable_terms = [UnfavorableTerm(**term_dict) for term_dict in unfavorable_terms]
        
        
        if missing_protections and isinstance(missing_protections[0], dict):
            
            missing_protections = [MissingProtection(**prot_dict) for prot_dict in missing_protections]
        
        return unfavorable_terms, missing_protections


    def _generate_protection_fallback(self, protection: MissingProtection) -> str:
        """
        Generate fallback for missing protections
        """
        if (protection.importance == "critical"):
            return "If they refuse, document this material gap and assess deal viability"
        
        else:
            return "If they refuse, note the gap and consider if other protections compensate"


    def _get_high_risk_categories(self, risk_analysis: RiskScore) -> List[str]:
        """
        Get high-risk categories from risk analysis
        """
        # Use the risk thresholds defined in RiskRules
        high_threshold = self.risk_rules.RISK_THRESHOLDS.get("high", 60)
        return [cat for cat, score in risk_analysis.category_scores.items() if (score >= high_threshold)]

    
    def _enhance_with_llm_strategy(self, points: List[NegotiationPoint], risk_analysis: RiskScore, provider: LLMProvider) -> List[NegotiationPoint]:
        """
        Use LLM to enhance negotiation points with sophisticated strategy
        """
        if not points:
            return points

        log_info(f"Enhancing {len(points)} negotiation points with LLM strategy")
        
        try:
            prompt   = self._create_strategic_enhancement_prompt(points, risk_analysis)
            response = self.llm_manager.complete(prompt             = prompt,
                                                 provider           = provider,
                                                 temperature        = 0.3,
                                                 max_tokens         = 2000,
                                                 fallback_providers = [LLMProvider.OPENAI],
                                                 retry_on_error     = True,
                                                )
            if response.success:
                enhanced = self._parse_strategic_enhancements(response.text, points)
                log_info("LLM strategic enhancement successful")
                return enhanced
            
            else:
                log_info("LLM strategic enhancement failed, using original points")
                return points
        
        except Exception as e:
            log_error(e, context = {"component": "NegotiationEngine", "operation": "enhance_with_llm_strategy"})
            return points

    
    def _create_strategic_enhancement_prompt(self, points: List[NegotiationPoint],  risk_analysis: RiskScore) -> str:
        """
        Create prompt for strategic LLM enhancement
        """
        context = {"overall_risk" : risk_analysis.overall_score,
                   "risk_level"   : risk_analysis.risk_level,
                   "points"       : [{"priority"   : p.priority,
                                      "issue"      : p.issue,
                                      "category"   : p.category,
                                      "current"    : p.current_language,
                                      "proposed"   : p.proposed_language,
                                      "tactic"     : p.tactic.value,
                                      "difficulty" : p.estimated_difficulty
                                     }
                                     for p in points
                                    ],
                  }

        prompt = f"""
                     As an expert negotiation strategist, enhance these negotiation points with sophisticated strategy.
                     CONTRACT RISK: {context['overall_risk']}/100 ({context['risk_level']})
                     NEGOTIATION POINTS (format: [{{'priority': int, 'issue': str, 'category': str, 'current': str, 'proposed': str, 'tactic': str, 'difficulty': str}}]):
                     {json.dumps(context['points'], indent=2)}
                     For EACH point (match the order and priority), provide:
                     1. ENHANCED_PROPOSAL: More specific, legally sound alternative language (only return the enhanced text).
                     2. STRATEGIC_RATIONALE: Business-focused reasoning emphasizing mutual benefit (only return the rationale).
                     3. COUNTERPARTY_PERSPECTIVE: Their likely concerns and how to address them (only return the perspective).
                     4. TIMING_STRATEGY: When and how to raise this issue (only return the timing).
                     5. BARGAINING_CHIPS: Specific trade-offs or concessions (only return the chips, as a comma-separated string).
                     Focus on creating win-win solutions and practical negotiation tactics. Respond in the exact format below for each point:
                     Point 1:
                     ENHANCED_PROPOSAL: [text]
                     STRATEGIC_RATIONALE: [text]
                     COUNTERPARTY_PERSPECTIVE: [text]
                     TIMING_STRATEGY: [text]
                     BARGAINING_CHIPS: [chip1, chip2, ...]
                     Point 2:
                     ENHANCED_PROPOSAL: [text]
                     ...
                  """

        return prompt


    def _parse_strategic_enhancements(self, llm_text: str, original_points: List[NegotiationPoint]) -> List[NegotiationPoint]:
        """
        Parse LLM strategic enhancements, assuming a structured format.
        """
        enhanced = list()

        for i, point in enumerate(original_points):
            point_identifier = f"Point {i+1}:"
            start_idx = llm_text.find(point_identifier)
            if (start_idx == -1):
                log_info(f"LLM response did not contain expected identifier for Point {i+1}. Keeping original.")
                enhanced.append(point)
                
                continue

            # Find the start of the next point or end of string
            next_point_idx = llm_text.find(f"Point {i+2}:", start_idx)
            if next_point_idx == -1:
                section_text = llm_text[start_idx:]
            
            else:
                section_text = llm_text[start_idx:next_point_idx]

            # Extract fields using regex within the section_text
            proposal_match = re.search(r"ENHANCED_PROPOSAL:\s*(.*?)(?:\n|$)", section_text, re.DOTALL)
            if proposal_match:
                new_proposal = proposal_match.group(1).strip()

                # Basic sanity check
                if (new_proposal and (len(new_proposal) > 10)): 
                    point.proposed_language = new_proposal 

            # TIMING_STRATEGY
            timing_match = re.search(r"TIMING_STRATEGY:\s*(.*?)(?:\n|$)", section_text, re.DOTALL)
            if timing_match:
                new_timing = timing_match.group(1).strip()
                if (new_timing and (len(new_timing) > 5)):
                    point.timing_suggestion = new_timing

            # BARGAINING_CHIPS
            chips_match = re.search(r"BARGAINING_CHIPS:\s*\[(.*?)\]", section_text, re.DOTALL)
            if chips_match:
                chips_str = chips_match.group(1).strip()
                
                if chips_str:
                    # Split by comma and strip whitespace
                    chips_list             = [chip.strip().strip('"\'') for chip in chips_str.split(',') if chip.strip()]

                    # Keep top 3
                    point.bargaining_chips = chips_list[:3] 

            enhanced.append(point)

        return enhanced


    def _generate_overall_strategy(self, risk_analysis: RiskScore, risk_interpretation: RiskInterpretation, contract_type: ContractType, provider: LLMProvider) -> str:
        """
        Generate overall negotiation strategy using LLM
        """
        prompt = f"""
                     As a negotiation expert, provide overall strategy for this contract.
                     CONTRACT TYPE: {contract_type.value}
                     RISK LEVEL: {risk_analysis.overall_score}/100 ({risk_analysis.risk_level})
                     KEY CONCERNS: {risk_interpretation.key_concerns}
                     Provide a concise 3-4 sentence negotiation strategy focusing on:
                     - Overall approach (collaborative vs. firm)
                     - Key priorities
                     - Risk management
                     - Success metrics
                     Strategy:
                  """
        try:
            response = self.llm_manager.complete(prompt      = prompt,
                                                 provider    = provider,
                                                 temperature = 0.3,
                                                 max_tokens  = 400,
                                                )
            
            return response.text.strip() if response.success else "Focus on addressing critical risks while maintaining collaborative negotiation tone."
        
        except Exception as e:
            log_error(e, context = {"operation": "generate_overall_strategy"})
            return "Prioritize critical risk items while seeking balanced, market-standard terms."


    def _identify_walk_away_items(self, negotiation_points: List[NegotiationPoint], risk_analysis: RiskScore) -> List[str]:
        """
        Identify non-negotiable walk-away items
        """
        walk_away       = list()
        critical_points = [p for p in negotiation_points if (p.priority == 1)]

        for point in critical_points:
            if ((point.estimated_difficulty == "hard") and (risk_analysis.overall_score >= 70)):
                walk_away.append(f"{point.issue} - critical risk that cannot be mitigated")

        # Max 5 walk-away items
        return walk_away[:5]  


    def _identify_concession_items(self, negotiation_points: List[NegotiationPoint], risk_analysis: RiskScore) -> List[str]:
        """
        Identify items that can be conceded
        """
        concessions  = list()
        low_priority = [p for p in negotiation_points if (p.priority >= 4)]

        for point in low_priority[:2]:
            if (point.estimated_difficulty == "hard"):
                concessions.append(f"{point.issue} - lower priority, high difficulty")

        return concessions


    def _generate_timing_guidance(self, negotiation_points: List[NegotiationPoint], contract_type: ContractType, provider: LLMProvider) -> str:
        """
        Generate timing guidance for negotiations
        """
        critical_count = len([p for p in negotiation_points if p.priority <= 2])
        
        if (critical_count >= 3):
            return "Start with critical items early - multiple deal-breakers need immediate attention"
        
        elif (critical_count >= 1):
            return "Address 1-2 critical items first, then move to high-priority items"
        
        else:
            return "Progressive approach: start with easier wins to build momentum"


    def _generate_risk_mitigation_plan(self, risk_analysis: RiskScore, negotiation_points: List[NegotiationPoint], provider: LLMProvider) -> str:
        """
        Generate risk mitigation plan for unresolved issues
        """
        if (risk_analysis.overall_score >= 70):
            return "High risk level - focus on critical term resolution. Have fallback positions ready."
        
        elif (risk_analysis.overall_score >= 50):
            return "Moderate risk - prioritize 2-3 key improvements. Document remaining risks."
        
        else:
            return "Manageable risk level - focus on most impactful improvements."
