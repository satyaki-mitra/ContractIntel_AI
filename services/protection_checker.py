@dataclass
class MissingProtection:
    """Missing protection item"""
    protection: str
    importance: str  # "critical", "high", "medium"
    explanation: str
    recommendation: str
    category: str


class ProtectionChecker:
    """Check for missing critical protections"""
    
    def __init__(self):
        self.rules = RiskRules()
    
    def check_missing_protections(self, contract_text: str, 
                                 clauses: List[ExtractedClause]) -> List[MissingProtection]:
        """Identify all missing protections"""
        
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
                    category=config["categories"][0]
                ))
        
        # Sort by importance
        importance_order = {"critical": 0, "high": 1, "medium": 2}
        missing.sort(key=lambda p: importance_order.get(p.importance, 3))
        
        return missing[:8]  # Top 8 missing protections
    
    def _check_protection(self, protection_id: str, text_lower: str, 
                         clauses: List[ExtractedClause]) -> bool:
        """Check if specific protection exists"""
        
        protection_indicators = {
            "for_cause_definition": ["for cause", "cause defined as", "grounds for termination include"],
            "severance_provision": ["severance", "severance pay", "separation package"],
            "mutual_indemnification": ["mutual indemnification", "each party shall indemnify", "both parties indemnify"],
            "liability_cap": ["liability.*cap", "maximum liability", "limited to.*fees", "not exceed"],
            "prior_ip_exclusion": ["prior.*intellectual property", "existing.*ip", "background.*ip", "pre-existing"],
            "confidentiality_duration": ["confidential.*period of", "for.*years from", "confidentiality.*expire"],
            "dispute_resolution": ["arbitration", "mediation", "dispute resolution"],
            "change_control_process": ["change order", "change request", "amendment.*writing", "modification.*writing"]
        }
        
        indicators = protection_indicators.get(protection_id, [])
        
        for indicator in indicators:
            if re.search(indicator, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _format_protection_name(self, protection_id: str) -> str:
        """Convert protection_id to readable name"""
        return protection_id.replace("_", " ").title()
    
    def _get_explanation(self, protection_id: str) -> str:
        """Get explanation for why this protection matters"""
        
        explanations = {
            "for_cause_definition": "Without a clear definition of 'for cause', termination grounds are ambiguous and could be abused.",
            "severance_provision": "No severance means zero compensation if terminated without cause, leaving you financially vulnerable.",
            "mutual_indemnification": "One-sided indemnification exposes you to liability without reciprocal protection from the other party.",
            "liability_cap": "Unlimited liability means unlimited financial risk, which is unreasonable for most professional services.",
            "prior_ip_exclusion": "Without excluding prior IP, they could claim ownership of your existing work and personal projects.",
            "confidentiality_duration": "Indefinite confidentiality is unreasonable and may restrict your future employment indefinitely.",
            "dispute_resolution": "Without dispute resolution procedures, any conflict could lead to costly litigation.",
            "change_control_process": "Without formal change procedures, scope creep and verbal modifications can create disputes."
        }
        
        return explanations.get(protection_id, "This protection is important for balanced risk allocation.")
    
    def _get_recommendation(self, protection_id: str) -> str:
        """Get recommendation for adding this protection"""
        
        recommendations = {
            "for_cause_definition": "Add: 'For Cause means: (a) gross negligence, (b) willful misconduct, (c) material breach after 30-day cure period'",
            "severance_provision": "Add: 'Upon termination without cause, Company shall pay [X months] salary as severance'",
            "mutual_indemnification": "Change to: 'Each party shall indemnify the other for losses arising from their respective breach or negligence'",
            "liability_cap": "Add: 'Total liability under this Agreement shall not exceed [12 months fees paid / $X amount]'",
            "prior_ip_exclusion": "Add: 'Work Product excludes Employee's prior intellectual property and personal projects unrelated to Company business'",
            "confidentiality_duration": "Add: 'Confidentiality obligations shall survive for [3-5] years after termination'",
            "dispute_resolution": "Add: 'Parties shall first attempt mediation before arbitration or litigation'",
            "change_control_process": "Add: 'All changes must be documented in written change orders signed by both parties'"
        }
        
        return recommendations.get(protection_id, "Negotiate to add this protection for balanced risk allocation.")

