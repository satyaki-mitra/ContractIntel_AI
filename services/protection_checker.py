"""
Missing Protections Checker
Identifies critical protections that should be in the contract but are missing
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.risk_rules import RiskRules
from services.clause_extractor import ExtractedClause
from utils.logger import ContractAnalyzerLogger, log_info


@dataclass
class MissingProtection:
    """Missing protection item"""
    protection: str
    importance: str  # "critical", "high", "medium"
    explanation: str
    recommendation: str
    category: str
    examples: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "protection": self.protection,
            "importance": self.importance,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "category": self.category,
            "examples": self.examples or []
        }


class ProtectionChecker:
    """Check for missing critical protections in contracts"""
    
    def __init__(self):
        self.rules = RiskRules()
        self.logger = ContractAnalyzerLogger.get_logger()
        log_info("ProtectionChecker initialized")
    
    @ContractAnalyzerLogger.log_execution_time("check_missing_protections")
    def check_missing_protections(self, contract_text: str, 
                                 clauses: List[ExtractedClause]) -> List[MissingProtection]:
        """
        Identify all missing protections
        
        Args:
            contract_text: Full contract text
            clauses: Extracted clauses
        
        Returns:
            List of MissingProtection objects
        """
        
        log_info("Starting missing protections check",
                text_length=len(contract_text),
                num_clauses=len(clauses))
        
        missing = []
        text_lower = contract_text.lower()
        
        for protection_id, config in self.rules.PROTECTION_CHECKLIST.items():
            is_present = self._check_protection(protection_id, text_lower, clauses)
            
            if not is_present:
                missing.append(MissingProtection(
                    protection=self._format_protection_name(protection_id),
                    importance=config["importance"],
                    explanation=self._get_explanation(protection_id),
                    recommendation=self._get_recommendation(protection_id),
                    category=config["categories"][0],
                    examples=self._get_examples(protection_id)
                ))
        
        # Sort by importance (critical > high > medium)
        importance_order = {"critical": 0, "high": 1, "medium": 2}
        missing.sort(key=lambda p: importance_order.get(p.importance, 3))
        
        log_info(f"Missing protections check complete",
                total_missing=len(missing),
                critical=sum(1 for p in missing if p.importance == "critical"))
        
        return missing[:10]  # Top 10 missing protections
    
    def _check_protection(self, protection_id: str, text_lower: str, 
                         clauses: List[ExtractedClause]) -> bool:
        """Check if specific protection exists"""
        
        protection_indicators = {
            "for_cause_definition": [
                "for cause", "cause defined as", "grounds for termination include",
                "cause means", "for cause includes"
            ],
            "severance_provision": [
                "severance", "severance pay", "separation package",
                "severance compensation", "separation payment"
            ],
            "mutual_indemnification": [
                "mutual indemnification", "each party shall indemnify",
                "both parties indemnify", "reciprocal indemnification"
            ],
            "liability_cap": [
                "liability.*cap", "maximum liability", "limited to.*fees",
                "not exceed", "liability shall not exceed"
            ],
            "prior_ip_exclusion": [
                "prior.*intellectual property", "existing.*ip",
                "background.*ip", "pre-existing", "prior inventions"
            ],
            "confidentiality_duration": [
                "confidential.*period of", "for.*years from",
                "confidentiality.*expire", "confidentiality.*term"
            ],
            "dispute_resolution": [
                "arbitration", "mediation", "dispute resolution",
                "resolution of disputes", "dispute mechanism"
            ],
            "change_control_process": [
                "change order", "change request", "amendment.*writing",
                "modification.*writing", "written consent"
            ]
        }
        
        indicators = protection_indicators.get(protection_id, [])
        
        # Check in full text
        for indicator in indicators:
            if indicator in text_lower:
                return True
        
        # Check in clauses (for more context-aware detection)
        relevant_categories = {
            "for_cause_definition": ["termination"],
            "severance_provision": ["termination", "compensation"],
            "mutual_indemnification": ["indemnification", "liability"],
            "liability_cap": ["liability"],
            "prior_ip_exclusion": ["intellectual_property"],
            "confidentiality_duration": ["confidentiality"],
            "dispute_resolution": ["dispute_resolution"],
            "change_control_process": ["amendment"]
        }
        
        categories = relevant_categories.get(protection_id, [])
        for clause in clauses:
            if clause.category in categories:
                for indicator in indicators:
                    if indicator in clause.text.lower():
                        return True
        
        return False
    
    def _format_protection_name(self, protection_id: str) -> str:
        """Convert protection_id to readable name"""
        names = {
            "for_cause_definition": "'For Cause' Definition",
            "severance_provision": "Severance Provision",
            "mutual_indemnification": "Mutual Indemnification",
            "liability_cap": "Liability Cap",
            "prior_ip_exclusion": "Prior IP Exclusion",
            "confidentiality_duration": "Confidentiality Duration Limit",
            "dispute_resolution": "Dispute Resolution Process",
            "change_control_process": "Change Control Process"
        }
        return names.get(protection_id, protection_id.replace("_", " ").title())
    
    def _get_explanation(self, protection_id: str) -> str:
        """Get explanation for why this protection matters"""
        
        explanations = {
            "for_cause_definition": (
                "Without a clear definition of 'for cause', termination grounds are ambiguous "
                "and could be abused. This leaves you vulnerable to arbitrary termination claims."
            ),
            "severance_provision": (
                "No severance means zero compensation if terminated without cause, leaving you "
                "financially vulnerable during job transition. Standard practice is 1-3 months salary."
            ),
            "mutual_indemnification": (
                "One-sided indemnification exposes you to liability without reciprocal protection "
                "from the other party. Fair contracts have mutual indemnification."
            ),
            "liability_cap": (
                "Unlimited liability means unlimited financial risk, which is unreasonable for most "
                "professional services. Standard practice is to cap liability at fees paid or a specific amount."
            ),
            "prior_ip_exclusion": (
                "Without excluding prior IP, they could claim ownership of your existing work and "
                "personal projects created before this agreement."
            ),
            "confidentiality_duration": (
                "Indefinite confidentiality is unreasonable and may restrict your future employment "
                "indefinitely. Standard practice is 3-5 years post-termination."
            ),
            "dispute_resolution": (
                "Without dispute resolution procedures, any conflict could lead to costly litigation. "
                "Mediation/arbitration clauses save time and money."
            ),
            "change_control_process": (
                "Without formal change procedures, scope creep and verbal modifications can create "
                "disputes about what was agreed. All changes should require written consent."
            )
        }
        
        return explanations.get(protection_id, "This protection is important for balanced risk allocation.")
    
    def _get_recommendation(self, protection_id: str) -> str:
        """Get recommendation for adding this protection"""
        
        recommendations = {
            "for_cause_definition": (
                "Add: 'For Cause means: (a) gross negligence or willful misconduct, "
                "(b) material breach of this Agreement after 30-day written cure period, "
                "(c) conviction of a felony, or (d) fraud or embezzlement.'"
            ),
            "severance_provision": (
                "Add: 'Upon termination without cause, Company shall pay Employee severance "
                "equal to [2-3] months of base salary, payable within 30 days.'"
            ),
            "mutual_indemnification": (
                "Change to: 'Each party shall indemnify and hold harmless the other party "
                "for losses arising from their respective breach, negligence, or willful misconduct.'"
            ),
            "liability_cap": (
                "Add: 'Total liability of either party under this Agreement shall not exceed "
                "the greater of (a) fees paid in the 12 months preceding the claim, or (b) $[amount].'"
            ),
            "prior_ip_exclusion": (
                "Add: 'Work Product excludes Employee's prior intellectual property, existing "
                "inventions, and personal projects unrelated to Company's business.'"
            ),
            "confidentiality_duration": (
                "Add: 'Confidentiality obligations shall survive termination for [3-5] years. "
                "Publicly available information is excluded from confidentiality.'"
            ),
            "dispute_resolution": (
                "Add: 'Disputes shall first be subject to good faith mediation. If unresolved "
                "after 30 days, either party may proceed to binding arbitration.'"
            ),
            "change_control_process": (
                "Add: 'All amendments and modifications must be in writing and signed by both parties. "
                "Verbal agreements are not binding.'"
            )
        }
        
        return recommendations.get(protection_id, "Negotiate to add this protection for balanced risk allocation.")
    
    def _get_examples(self, protection_id: str) -> List[str]:
        """Get example language for this protection"""
        
        examples = {
            "for_cause_definition": [
                "\"For Cause\" means (a) gross negligence, (b) willful misconduct, (c) material breach after cure period",
                "Termination for cause requires written notice specifying the grounds",
                "Employee has 30 days to cure any alleged breach before termination"
            ],
            "severance_provision": [
                "2-3 months base salary as severance for termination without cause",
                "Severance payable within 30 days of termination date",
                "Prorated bonus included in severance calculation"
            ],
            "mutual_indemnification": [
                "Each party indemnifies the other for their own negligence",
                "Indemnification is mutual and reciprocal",
                "Liability capped for both parties equally"
            ],
            "liability_cap": [
                "Liability capped at 12 months of fees paid",
                "No liability for consequential or indirect damages",
                "Cap applies to both parties equally"
            ],
            "prior_ip_exclusion": [
                "Prior inventions listed in Exhibit A are excluded",
                "Personal projects unrelated to Company business excluded",
                "Background IP remains Employee's property"
            ],
            "confidentiality_duration": [
                "Confidentiality obligations expire 5 years after termination",
                "Publicly available information excluded from confidentiality",
                "Trade secrets protected indefinitely but defined specifically"
            ],
            "dispute_resolution": [
                "Mediation required before arbitration or litigation",
                "Arbitration in neutral location with shared costs",
                "Winner recovers reasonable attorneys' fees"
            ],
            "change_control_process": [
                "All changes require written change order signed by both parties",
                "Verbal agreements are not binding",
                "Change orders must specify scope, cost, and timeline"
            ]
        }
        
        return examples.get(protection_id, [])
    
    def get_critical_missing(self, protections: List[MissingProtection]) -> List[MissingProtection]:
        """Filter to only critical missing protections"""
        critical = [p for p in protections if p.importance == "critical"]
        log_info(f"Found {len(critical)} critical missing protections")
        return critical
    
    def get_by_category(self, protections: List[MissingProtection], 
                       category: str) -> List[MissingProtection]:
        """Filter protections by category"""
        filtered = [p for p in protections if p.category == category]
        log_info(f"Found {len(filtered)} missing protections in category '{category}'")
        return filtered
    
    def get_importance_distribution(self, protections: List[MissingProtection]) -> Dict[str, int]:
        """Get distribution by importance level"""
        distribution = {"critical": 0, "high": 0, "medium": 0}
        
        for protection in protections:
            distribution[protection.importance] = distribution.get(protection.importance, 0) + 1
        
        log_info("Importance distribution", **distribution)
        
        return distribution