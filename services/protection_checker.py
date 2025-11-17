# DEPENDENCIES
import re
import sys
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
from services.data_models import ExtractedClause
from services.data_models import MissingProtection


class ProtectionChecker:
    """
    Check for missing critical protections in contracts using RiskRules framework
    """
    def __init__(self, contract_type: ContractType = ContractType.GENERAL):
        """
        Initialize protection checker with contract-type specific analysis

        Arguments:
        ----------
            contract_type { ContractType } : Contract type for protection prioritization
        """
        self.contract_type         = contract_type
        self.rules                 = RiskRules()
        self.logger                = ContractAnalyzerLogger.get_logger()

        # Contract-type specific protection priorities
        self.protection_priorities = self._get_contract_type_priorities()

        log_info("ProtectionChecker initialized",
                 contract_type    = self.contract_type.value,
                 protection_count = len(self.rules.PROTECTION_CHECKLIST),
                )


    def _get_contract_type_priorities(self) -> Dict[str, List[str]]:
        """
        Get protection priorities by contract type
        """
        priorities = {ContractType.EMPLOYMENT.value : ['for_cause_definition', 'severance_provision', 'prior_ip_exclusion', 'confidentiality_duration'],
                      ContractType.SOFTWARE.value   : ['liability_cap', 'prior_ip_exclusion', 'mutual_indemnification', 'dispute_resolution'],
                      ContractType.CONSULTING.value : ['liability_cap', 'mutual_indemnification', 'payment_terms', 'change_control_process'],
                      ContractType.NDA.value        : ['confidentiality_duration', 'prior_ip_exclusion', 'dispute_resolution'],
                      ContractType.LEASE.value      : ['dispute_resolution', 'change_control_process', 'insurance_requirements'],
                      ContractType.PURCHASE.value   : ['liability_cap', 'warranty_protection', 'dispute_resolution'],
                      ContractType.GENERAL.value    : ['liability_cap', 'mutual_indemnification', 'dispute_resolution'],
                     }

        return priorities.get(self.contract_type.value, [])


    @ContractAnalyzerLogger.log_execution_time("check_missing_protections")
    def check_missing_protections(self, contract_text: str, clauses: List[ExtractedClause], contract_type: Optional[ContractType] = None) -> List[MissingProtection]:
        """
        Identify all missing protections using comprehensive RiskRules framework

        Arguments:
        ----------
            contract_text { str }          : Full contract text

            clauses       { list }         : Extracted clauses

            contract_type { ContractType } : Override contract type

        Returns:
        --------
                      { list }             : List of MissingProtection objects
        """

        # Update contract type if provided
        if contract_type:
            self.contract_type         = contract_type
            self.protection_priorities = self._get_contract_type_priorities()

        log_info("Starting missing protections analysis",
                 text_length   = len(contract_text),
                 num_clauses   = len(clauses),
                 contract_type = self.contract_type.value,
                )

        missing    = list()
        text_lower = contract_text.lower()

        # Check each protection in RiskRules PROTECTION_CHECKLIST
        for protection_id, config in self.rules.PROTECTION_CHECKLIST.items():
            is_present, found_in_clauses = self._check_protection_comprehensive(protection_id = protection_id,
                                                                                text_lower    = text_lower,
                                                                                clauses       = clauses,
                                                                               )

            if not is_present:
                missing_protection = self._create_missing_protection(protection_id    = protection_id,
                                                                     config           = config,
                                                                     found_in_clauses = found_in_clauses,
                                                                    )

                missing.append(missing_protection)

        # Prioritize by contract type and risk score
        final_missing = self._prioritize_missing_protections(missing_protections = missing)

        log_info("Missing protections analysis complete",
                 total_missing = len(final_missing),
                 critical      = sum(1 for p in final_missing if (p.importance == "critical")),
                 high          = sum(1 for p in final_missing if (p.importance == "high")),
                )

        return final_missing


    def _check_protection_comprehensive(self, protection_id: str, text_lower: str, clauses: List[ExtractedClause]) -> Tuple[bool, List[str]]:
        """
        Comprehensive protection detection using multiple methods

        Returns:
        --------
            { tuple } : (is_present, list of clause references where protection was found)
        """
        found_in_clauses    = list()

        # Enhanced protection patterns with regex for better matching
        protection_patterns = self._get_protection_patterns(protection_id = protection_id)

        # Check in full text with regex patterns
        for pattern in protection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, found_in_clauses

        # Check in relevant clauses with context awareness
        relevant_categories = self.rules.PROTECTION_CHECKLIST[protection_id]["categories"]
        relevant_clauses    = [c for c in clauses if c.category in relevant_categories]

        for clause in relevant_clauses:
            clause_text_lower = clause.text.lower()

            for pattern in protection_patterns:
                if re.search(pattern, clause_text_lower, re.IGNORECASE):
                    found_in_clauses.append(clause.reference)
                    
                    return True, found_in_clauses

        # Additional semantic checks for complex protections
        if self._check_protection_semantic(protection_id=protection_id, text_lower=text_lower, clauses=clauses):
            return True, found_in_clauses

        return False, found_in_clauses


    def _get_protection_patterns(self, protection_id: str) -> List[str]:
        """
        Get comprehensive regex patterns for each protection
        """
        patterns = {"for_cause_definition"     : [r'for\s+cause\s+means', r'cause\s+defined\s+as', r'grounds?\s+for\s+termination', r'termination\s+for\s+cause', r'just\s+cause\s+definition',],
                    "severance_provision"      : [r'severance\s+(pay|compensation|benefits)', r'separation\s+(pay|package|compensation)', r'termination\s+(pay|benefits)', r'upon\s+termination.*pay', r'severance.*equal\s+to',],
                    "mutual_indemnification"   : [r'mutual\s+indemnification', r'each\s+party\s+shall\s+indemnify', r'both\s+parties\s+indemnify', r'reciprocal\s+indemnification', r'indemnification.*mutual',],
                    "liability_cap"            : [r'liability.*cap', r'maximum\s+liability', r'limited\s+to.*\$?\d+', r'not\s+exceed.*\$?\d+', r'liability\s+shall\s+not\s+exceed', r'cap.*liability',],
                    "prior_ip_exclusion"       : [r'prior\s+intellectual\s+property', r'existing\s+ip', r'background\s+ip', r'pre-existing', r'prior\s+inventions', r'personal\s+projects',],
                    "confidentiality_duration" : [r'confidentiality.*period\s+of', r'for\s+\d+\s+years\s+from', r'confidentiality.*expire', r'confidentiality.*term', r'duration.*confidentiality',],
                    "dispute_resolution"       : [r'arbitration', r'mediation', r'dispute\s+resolution', r'resolution\s+of\s+disputes', r'alternative\s+dispute', r'adr',],
                    "change_control_process"   : [r'change\s+order', r'change\s+request', r'amendment.*writing', r'modification.*writing', r'written\s+consent', r'change\s+control',],
                    "insurance_requirements"   : [r'insurance\s+requirements', r'maintain\s+insurance', r'proof\s+of\s+insurance', r'coverage.*\$?\d+', r'liability\s+insurance',],
                    "force_majeure"            : [ r'force\s+majeure', r'act\s+of\s+god', r'unforeseeable', r'beyond\s+control', r'natural\s+disaster',],
                   }

        return patterns.get(protection_id, [rf'\b{protection_id}\b'])


    def _check_protection_semantic(self, protection_id: str, text_lower: str, clauses: List[ExtractedClause]) -> bool:
        """
        Semantic checks for complex protections that need context understanding
        """
        if (protection_id == "mutual_indemnification"):
            # Check if there's any indemnification that's not mutual
            has_indemnification = bool(re.search(r'indemnif', text_lower))
            has_mutual_language = bool(re.search(r'mutual|each party|both parties', text_lower))

            return has_indemnification and has_mutual_language

        elif (protection_id == "liability_cap"):
            # Check if there's liability language but no cap
            has_liability = bool(re.search(r'liability|liable', text_lower))
            has_cap       = bool(re.search(r'cap|limit|maximum|not exceed', text_lower))

            return has_liability and has_cap

        elif (protection_id == "prior_ip_exclusion"):
            # Check if there's IP assignment but no exclusion
            has_ip_assignment = bool(re.search(r'intellectual property|work product|inventions', text_lower))
            has_exclusion     = bool(re.search(r'prior|existing|background|exclude', text_lower))

            return has_ip_assignment and has_exclusion

        return False


    def _create_missing_protection(self, protection_id: str, config: Dict, found_in_clauses: List[str]) -> MissingProtection:
        """
        Create comprehensive MissingProtection object
        """
        # Use centralized map for display name
        protection_name = self.rules.get_protection_display_name(protection_id)

        return MissingProtection(protection_id      = protection_id, 
                                 protection         = protection_name,
                                 importance         = config["importance"],
                                 risk_score         = config["risk_if_missing"],
                                 explanation        = self._get_comprehensive_explanation(protection_id = protection_id),
                                 recommendation     = self._get_detailed_recommendation(protection_id = protection_id),
                                 categories         = config["categories"],
                                 contract_type      = self.contract_type.value,
                                 suggested_language = self._get_suggested_language(protection_id = protection_id),
                                 legal_basis        = self._get_legal_basis(protection_id = protection_id),
                                 affected_clauses   = found_in_clauses,
                                )


    def _get_comprehensive_explanation(self, protection_id: str) -> str:
        """
        Get detailed explanation for why this protection matters
        """
        explanations = {"for_cause_definition"     : ("Without a clear 'for cause' definition, termination grounds remain ambiguous and subject to interpretation abuse. "
                                                      "This creates significant job insecurity and potential for arbitrary termination without proper recourse."
                                                     ),
                        "severance_provision"      : ("Missing severance provision means zero financial protection if terminated without cause. "
                                                      "Industry standards provide 2-3 months salary to support transition and mitigate sudden income loss."
                                                     ),
                        "mutual_indemnification"   : ("One-sided indemnification creates asymmetric liability exposure. Mutual protection ensures both parties share "
                                                      "responsibility for their respective breaches, negligence, or misconduct."
                                                     ),
                        "liability_cap"            : ("Unlimited liability exposes you to catastrophic financial risk beyond reasonable business expectations. "
                                                      "Standard practice caps liability at fees paid or a reasonable multiple of contract value."
                                                     ),
                        "prior_ip_exclusion"       : ("Without prior IP exclusion, your existing intellectual property and personal projects could be claimed by the other party. "
                                                      "This protection preserves ownership of work created before and outside this engagement."
                                                     ),
                        "confidentiality_duration" : ("Indefinite confidentiality obligations unreasonably restrict future business activities indefinitely. "
                                                      "Industry standards limit confidentiality to 3-5 years post-termination for most information."
                                                     ),
                        "dispute_resolution"       : ("Without formal dispute resolution, conflicts escalate directly to costly litigation. Mediation and arbitration "
                                                      "provide efficient, cost-effective alternatives with specialized expertise."
                                                     ),
                        "change_control_process"   : ("Lack of change control enables scope creep and verbal modifications that create ambiguity. Formal processes "
                                                      "ensure all changes are documented, approved, and properly scoped."
                                                     ),
                        "insurance_requirements"   : ("Missing insurance requirements leave you exposed to uncovered liabilities. "
                                                      "Proper coverage transfers risk and provides financial protection for both parties."
                                                     ),
                        "force_majeure"            : ("Without force majeure protection, you remain liable for performance during unforeseeable events beyond control. "
                                                      "This clause provides reasonable relief during extraordinary circumstances."
                                                     ),
                       }

        return explanations.get(protection_id, "This protection is critical for balanced risk allocation and legal fairness.")


    def _get_detailed_recommendation(self, protection_id: str) -> str:
        """
        Get detailed recommendation for adding this protection
        """
        recommendations = {"for_cause_definition"     : ("Add clear 'For Cause' definition including: gross negligence, willful misconduct, material breach after "
                                                         "30-day cure period, conviction of felony, or fraud. Require written notice specifying grounds."
                                                        ),
                           "severance_provision"      : ("Include severance equal to 2-3 months base salary for termination without cause, payable within 30 days. "
                                                         "Add pro-rated bonus calculation and continuation of benefits during severance period."
                                                        ),
                           "mutual_indemnification"   : ("Replace one-sided language with: 'Each party shall indemnify, defend, and hold harmless the other party "
                                                         "from claims arising from their respective breach, negligence, or willful misconduct.'"
                                                        ),
                           "liability_cap"            : ("Add: 'Total liability of either party under this Agreement shall not exceed the greater of (a) fees paid "
                                                         "in the 12 months preceding the claim, or (b) $[reasonable amount]. Exclude liability for indirect damages.'"
                                                        ),
                           "prior_ip_exclusion"       : ("Include: 'Work Product excludes Employee's prior intellectual property, existing inventions, personal projects "
                                                         "unrelated to Company business, and open source contributions. Attach prior IP list as Exhibit A.'"
                                                        ),
                           "confidentiality_duration" : ("Specify: 'Confidentiality obligations shall survive termination for 3-5 years. Trade secrets protected "
                                                         "indefinitely but must be specifically identified. Publicly available information excluded.'"
                                                        ),
                           "dispute_resolution"       : ("Add: 'Disputes shall first be subject to 30-day good faith mediation. If unresolved, binding arbitration "
                                                         "under [rules] in [neutral location]. Each party bears own costs, arbitrator may award fees to prevailing party.'"
                                                        ),
                           "change_control_process"   : ("Include: 'All amendments require written change orders signed by both parties. Change orders must specify "
                                                         "scope, timeline, cost, and acceptance criteria. Verbal agreements are not binding.'"
                                                        ),
                           "insurance_requirements"   : ("Specify: 'Contractor shall maintain general liability insurance of $1M per occurrence, professional liability "
                                                         "insurance of $2M, and workers' compensation. Provide certificates of insurance before commencement.'"
                                                        ),
                           "force_majeure"            : ("Add: 'Neither party liable for failure to perform due to causes beyond reasonable control including acts of God, "
                                                         "war, strikes, or natural disasters. Performance suspended during event, resume when practicable.'"
                                                        ),
                          }

        return recommendations.get(protection_id, "Negotiate to include this standard protection for balanced risk allocation.")


    def _get_suggested_language(self, protection_id: str) -> str:
        """
        Get actual suggested clause language
        """
        language_library = {"for_cause_definition"     : ("\"For Cause\" means: (a) gross negligence or willful misconduct; (b) material breach of this Agreement after 30-day written notice and cure period; (c) conviction of a felony; or (d) fraud, dishonesty, or embezzlement."),
                            "severance_provision"      : ("Upon termination without cause, Company shall pay Employee severance equal to three months of base salary, payable within 30 days of termination. Employee shall also receive pro-rated annual bonus and continuation of health benefits during severance period."),
                            "mutual_indemnification"   : ("Each party shall indemnify, defend, and hold harmless the other party from and against any and all claims, damages, losses, and expenses arising from the indemnifying party's breach of this Agreement, negligence, or willful misconduct."),
                            "liability_cap"            : ("Notwithstanding anything to the contrary, the total liability of either party under this Agreement shall not exceed the greater of (a) the fees paid by Customer to Provider in the twelve months preceding the claim, or (b) $500,000. Neither party shall be liable for any indirect, special, incidental, or consequential damages."),
                            "prior_ip_exclusion"       : ("Work Product excludes any intellectual property, inventions, or creative works developed by Employee prior to this Agreement or developed outside the scope of employment without using Company resources. Employee has listed prior IP in Exhibit A. Background IP remains the property of its respective owner."),
                            "confidentiality_duration" : ("The obligations of confidentiality shall survive termination of this Agreement for a period of five years. Trade secrets shall be protected indefinitely. Confidential Information shall not include information that is or becomes publicly available through no fault of Receiving Party."),
                            "dispute_resolution"       : ("Any dispute arising under this Agreement shall first be submitted to mediation with a mutually acceptable mediator. If mediation fails after 30 days, either party may initiate binding arbitration under the rules of the American Arbitration Association. The prevailing party in any dispute shall be entitled to recover reasonable attorneys' fees and costs."),
                            "change_control_process"   : ("No amendment, modification, or waiver of any provision of this Agreement shall be effective unless in writing and signed by both parties. All change requests must be submitted in writing as Change Orders, specifying the changes, associated costs, timeline impacts, and acceptance criteria."),
                            "insurance_requirements"   : ("Contractor shall maintain at its own expense: (a) Commercial General Liability insurance with limits of $1,000,000 per occurrence; (b) Professional Liability insurance with limits of $2,000,000 per claim; and (c) Workers' Compensation insurance as required by law. Certificates of insurance shall be provided to Client upon request."),
                            "force_majeure"            : ("Neither party shall be liable for any failure or delay in performance under this Agreement due to causes beyond its reasonable control, including acts of God, war, terrorism, labor disputes, or governmental actions. The affected party shall notify the other party promptly and resume performance as soon as practicable."),
                           }

        return language_library.get(protection_id, "Standard protection clause appropriate for this contract type.")


    def _get_legal_basis(self, protection_id: str) -> str:
        """
        Get legal basis for why this protection is important
        """
        legal_bases = {"for_cause_definition"     : "Employment protection statutes and doctrine of good faith and fair dealing",
                       "severance_provision"      : "Industry standards and reasonable notice requirements",
                       "mutual_indemnification"   : "Principle of mutuality and unconscionability doctrine",
                       "liability_cap"            : "Commercial reasonableness and risk allocation principles",
                       "prior_ip_exclusion"       : "Intellectual property rights and prior ownership protection",
                       "confidentiality_duration" : "Reasonableness standard for restrictive covenants",
                       "dispute_resolution"       : "Efficient dispute resolution and access to justice",
                       "change_control_process"   : "Contract formation and modification requirements",
                       "insurance_requirements"   : "Risk management and liability transfer principles",
                       "force_majeure"            : "Impossibility of performance and commercial impracticability",
                      }

        return legal_bases.get(protection_id, "Standard contractual protection for balanced risk allocation")


    def _prioritize_missing_protections(self, missing_protections: List[MissingProtection]) -> List[MissingProtection]:
        """
        Prioritize missing protections by contract type and risk score
        """
        if not missing_protections:
            return []

        # Sort by risk score (descending)
        missing_protections.sort(key = lambda p: p.risk_score, reverse = True)

        # Boost priority for contract-type specific critical protections
        for protection in missing_protections:
            # Use the protection_id for the check
            if protection.protection_id in self.protection_priorities:
                # Boost for contract relevance
                protection.risk_score += 10

        # Re-sort with boosted scores
        missing_protections.sort(key = lambda p: p.risk_score, reverse = True)
        

        # Return top 15 most critical missing protections
        top_missing_protections = missing_protections[:15]
        
        return top_missing_protections


    def get_critical_missing(self, protections: List[MissingProtection]) -> List[MissingProtection]:
        """
        Filter to only critical missing protections
        """
        critical = [p for p in protections if (p.importance == "critical")]

        log_info(f"Found {len(critical)} critical missing protections")

        return critical


    def get_by_category(self, protections: List[MissingProtection], category: str) -> List[MissingProtection]:
        """
        Filter protections by category
        """
        filtered = [p for p in protections if category in p.categories]

        log_info(f"Found {len(filtered)} missing protections in category '{category}'")

        return filtered


    def get_importance_distribution(self, protections: List[MissingProtection]) -> Dict[str, int]:
        """
        Get distribution by importance level
        """
        distribution = {"critical" : 0, 
                        "high"     : 0, 
                        "medium"   : 0, 
                        "low"      : 0,
                       }

        for protection in protections:
            distribution[protection.importance] = distribution.get(protection.importance, 0) + 1

        log_info("Missing protections importance distribution", **distribution)

        return distribution


    def get_risk_score_summary(self, protections: List[MissingProtection]) -> Dict[str, float]:
        """
        Get risk score summary statistics
        """
        if not protections:
            return {"total_risk"   : 0, 
                    "average_risk" : 0, 
                    "max_risk"     : 0,
                   }

        scores       = [p.risk_score for p in protections]
        total_risk   = sum(scores)
        average_risk = total_risk / len(scores)
        max_risk     = max(scores)

        summary      = {"total_risk"   : round(total_risk, 2),
                        "average_risk" : round(average_risk, 2),
                        "max_risk"     : round(max_risk, 2),
                       }

        log_info("Missing protections risk score summary", **summary)

        return summary
