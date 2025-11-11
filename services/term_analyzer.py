"""
Unfavorable Terms Analyzer
Detects one-sided, punitive, and unfavorable contract terms
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.clause_extractor import ExtractedClause
from utils.logger import ContractAnalyzerLogger, log_info, log_error


@dataclass
class UnfavorableTerm:
    """Detected unfavorable term"""
    term: str
    category: str
    severity: str  # "critical", "high", "medium"
    explanation: str
    clause_reference: Optional[str] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "term": self.term,
            "category": self.category,
            "severity": self.severity,
            "explanation": self.explanation,
            "clause_reference": self.clause_reference,
            "suggested_fix": self.suggested_fix
        }


class TermAnalyzer:
    """Detect unfavorable and one-sided terms in contracts"""
    
    def __init__(self):
        self.logger = ContractAnalyzerLogger.get_logger()
        log_info("TermAnalyzer initialized")
    
    @ContractAnalyzerLogger.log_execution_time("analyze_unfavorable_terms")
    def analyze_unfavorable_terms(self, contract_text: str, 
                                 clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Detect all unfavorable terms in contract
        
        Args:
            contract_text: Full contract text
            clauses: Extracted clauses
        
        Returns:
            List of UnfavorableTerm objects
        """
        
        log_info("Starting unfavorable terms analysis",
                text_length=len(contract_text),
                num_clauses=len(clauses))
        
        unfavorable_terms = []
        
        # Check each clause for unfavorable patterns
        for clause in clauses:
            terms = self._analyze_clause_terms(clause)
            unfavorable_terms.extend(terms)
        
        # Check full text for cross-clause issues
        cross_clause_terms = self._analyze_cross_clause_issues(contract_text, clauses)
        unfavorable_terms.extend(cross_clause_terms)
        
        # Deduplicate and prioritize
        final_terms = self._deduplicate_and_prioritize(unfavorable_terms)
        
        log_info(f"Unfavorable terms analysis complete",
                total_found=len(final_terms),
                critical=sum(1 for t in final_terms if t.severity == "critical"))
        
        return final_terms
    
    def _analyze_clause_terms(self, clause: ExtractedClause) -> List[UnfavorableTerm]:
        """Analyze a single clause for unfavorable terms"""
        terms = []
        text_lower = clause.text.lower()
        
        # Critical unfavorable patterns
        critical_patterns = {
            'unlimited_liability': {
                'pattern': r'unlimited.*liability|liability.*unlimited',
                'explanation': "Unlimited liability exposes you to potentially catastrophic financial risk with no cap.",
                'fix': "Add: 'Liability shall be capped at [amount] or fees paid in the preceding 12 months.'"
            },
            'perpetual_obligation': {
                'pattern': r'perpetual|indefinite.*obligation|forever.*bound',
                'explanation': "Perpetual obligations never expire, binding you indefinitely even after termination.",
                'fix': "Specify a reasonable time period (e.g., 3-5 years) for post-termination obligations."
            },
            'wage_withholding': {
                'pattern': r'(may|can|shall)\s+(withhold|deduct|retain).*compensation',
                'explanation': "Wage withholding clauses are likely illegal in most jurisdictions and highly unfavorable.",
                'fix': "Remove this clause entirely. Wage withholding is generally prohibited by labor law."
            },
            'liquidated_damages': {
                'pattern': r'liquidated\s+damages.*\$?\d+',
                'explanation': "Liquidated damages can be punitive penalties rather than reasonable compensation.",
                'fix': "Ensure damages are a reasonable estimate of actual harm, not a penalty."
            }
        }
        
        for term_id, config in critical_patterns.items():
            if re.search(config['pattern'], text_lower, re.IGNORECASE):
                terms.append(UnfavorableTerm(
                    term=term_id.replace('_', ' ').title(),
                    category=clause.category,
                    severity="critical",
                    explanation=config['explanation'],
                    clause_reference=clause.reference,
                    suggested_fix=config['fix']
                ))
        
        # High-risk unfavorable patterns
        high_risk_patterns = {
            'sole_discretion': {
                'pattern': r'(sole|absolute|unfettered)\s+discretion',
                'explanation': "Sole discretion clauses give one party unilateral power without accountability or standards.",
                'fix': "Change to 'reasonable discretion' or 'mutual agreement' to ensure fairness."
            },
            'at_will_termination': {
                'pattern': r'at-will|at\s+will.*terminat|terminat.*without\s+cause',
                'explanation': "At-will termination allows immediate firing without cause or severance, creating job insecurity.",
                'fix': "Add 'for cause' definition and require notice period with severance for termination without cause."
            },
            'non_compete_overly_broad': {
                'pattern': r'(entire|all|worldwide|global)\s*(industry|market|territory)',
                'explanation': "Overly broad non-compete clauses unreasonably restrict your ability to earn a living.",
                'fix': "Limit to direct competitors within a specific geographic area for a reasonable time (6-12 months)."
            },
            'immediate_termination': {
                'pattern': r'immediate.*terminat|terminat.*immediate',
                'explanation': "Immediate termination without notice or cure period is harsh and one-sided.",
                'fix': "Require written notice and 30-day cure period before termination for most breaches."
            }
        }
        
        for term_id, config in high_risk_patterns.items():
            if re.search(config['pattern'], text_lower, re.IGNORECASE):
                terms.append(UnfavorableTerm(
                    term=term_id.replace('_', ' ').title(),
                    category=clause.category,
                    severity="high",
                    explanation=config['explanation'],
                    clause_reference=clause.reference,
                    suggested_fix=config['fix']
                ))
        
        # Medium-risk unfavorable patterns
        medium_risk_patterns = {
            'vague_compensation': {
                'pattern': r'(to be determined|tbd|subject to review).*compensation',
                'explanation': "Vague compensation terms leave you vulnerable to arbitrary or unfair pay decisions.",
                'fix': "Specify exact amounts, formulas, or clear criteria for determining compensation."
            },
            'unequal_obligations': {
                'pattern': r'employee\s+shall.*must.*but.*employer\s+may',
                'explanation': "One party has mandatory obligations ('shall') while the other has optional rights ('may').",
                'fix': "Balance obligations or make key provisions mutual with 'both parties shall'."
            },
            'auto_renewal': {
                'pattern': r'(automatically|auto).*renew',
                'explanation': "Auto-renewal clauses can trap you in unfavorable agreements without conscious re-commitment.",
                'fix': "Require explicit written consent for renewals or add opt-out period before renewal."
            }
        }
        
        for term_id, config in medium_risk_patterns.items():
            if re.search(config['pattern'], text_lower, re.IGNORECASE):
                terms.append(UnfavorableTerm(
                    term=term_id.replace('_', ' ').title(),
                    category=clause.category,
                    severity="medium",
                    explanation=config['explanation'],
                    clause_reference=clause.reference,
                    suggested_fix=config['fix']
                ))
        
        return terms
    
    def _analyze_cross_clause_issues(self, text: str, 
                                    clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """Detect issues that span multiple clauses"""
        terms = []
        
        # Check for imbalanced notice periods
        notice_imbalance = self._check_notice_imbalance(clauses)
        if notice_imbalance:
            terms.append(notice_imbalance)
        
        # Check for missing reciprocal provisions
        missing_reciprocal = self._check_missing_reciprocal(text, clauses)
        terms.extend(missing_reciprocal)
        
        # Check for conflicting clauses
        conflicts = self._check_conflicting_clauses(clauses)
        terms.extend(conflicts)
        
        return terms
    
    def _check_notice_imbalance(self, clauses: List[ExtractedClause]) -> Optional[UnfavorableTerm]:
        """Check if notice periods are imbalanced between parties"""
        term_clauses = [c for c in clauses if c.category == "termination"]
        
        if not term_clauses:
            return None
        
        # Extract notice periods
        text = " ".join([c.text for c in term_clauses])
        notice_pattern = r'(\d+)\s*days?\s*(notice|prior\s+notice)'
        matches = re.findall(notice_pattern, text, re.IGNORECASE)
        
        if len(matches) >= 2:
            periods = [int(m[0]) for m in matches]
            ratio = max(periods) / min(periods)
            
            if ratio >= 2:
                severity = "critical" if ratio >= 3 else "high"
                return UnfavorableTerm(
                    term="Imbalanced Notice Periods",
                    category="termination",
                    severity=severity,
                    explanation=f"Notice period imbalance: {max(periods)} days vs {min(periods)} days (ratio: {ratio:.1f}x). This creates unfair burden on one party.",
                    clause_reference=term_clauses[0].reference,
                    suggested_fix=f"Equalize notice periods to {min(periods)} days for both parties, or make them mutually reasonable (e.g., 30 days mutual)."
                )
        
        return None
    
    def _check_missing_reciprocal(self, text: str, 
                                 clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """Check for missing reciprocal provisions"""
        terms = []
        text_lower = text.lower()
        
        # Check indemnification reciprocity
        indem_clauses = [c for c in clauses if c.category == "indemnification"]
        if indem_clauses:
            has_one_sided = any(
                re.search(r'(you|employee|consultant|contractor)\s+shall\s+indemnify', c.text, re.IGNORECASE)
                for c in indem_clauses
            )
            has_mutual = any(
                "mutual" in c.text.lower() or "both parties" in c.text.lower() or "each party" in c.text.lower()
                for c in indem_clauses
            )
            
            if has_one_sided and not has_mutual:
                terms.append(UnfavorableTerm(
                    term="Non-Reciprocal Indemnification",
                    category="indemnification",
                    severity="critical",
                    explanation="You must indemnify them, but there's no reciprocal protection. This creates one-sided liability exposure.",
                    clause_reference=indem_clauses[0].reference,
                    suggested_fix="Change to mutual indemnification: 'Each party shall indemnify the other for losses arising from their respective breach or negligence.'"
                ))
        
        # Check IP assignment reciprocity
        ip_clauses = [c for c in clauses if c.category == "intellectual_property"]
        if ip_clauses:
            assigns_all = any(
                "all" in c.text.lower() and "work product" in c.text.lower()
                for c in ip_clauses
            )
            excludes_prior = any(
                "prior" in c.text.lower() or "existing" in c.text.lower() or "background" in c.text.lower()
                for c in ip_clauses
            )
            
            if assigns_all and not excludes_prior:
                terms.append(UnfavorableTerm(
                    term="Overly Broad IP Assignment",
                    category="intellectual_property",
                    severity="high",
                    explanation="Assigns all IP without excluding prior/personal work. Could claim ownership of your existing projects.",
                    clause_reference=ip_clauses[0].reference,
                    suggested_fix="Add: 'Work Product excludes Employee's prior intellectual property, personal projects, and inventions unrelated to Company business.'"
                ))
        
        # Check liability cap reciprocity
        liab_clauses = [c for c in clauses if c.category == "liability"]
        if liab_clauses:
            has_cap = any(
                "cap" in c.text.lower() or "limited to" in c.text.lower()
                for c in liab_clauses
            )
            has_unlimited = any(
                "unlimited" in c.text.lower() or "no limit" in c.text.lower()
                for c in liab_clauses
            )
            
            if has_unlimited or not has_cap:
                terms.append(UnfavorableTerm(
                    term="No Liability Cap",
                    category="liability",
                    severity="critical",
                    explanation="Unlimited or uncapped liability exposes you to potentially unlimited financial damages.",
                    clause_reference=liab_clauses[0].reference if liab_clauses else None,
                    suggested_fix="Add mutual liability cap: 'Total liability of either party shall not exceed [amount] or fees paid in preceding 12 months.'"
                ))
        
        return terms
    
    def _check_conflicting_clauses(self, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """Detect conflicting or contradictory clauses"""
        terms = []
        
        # Check for termination conflicts
        term_clauses = [c for c in clauses if c.category == "termination"]
        if len(term_clauses) >= 2:
            # Look for conflicting termination provisions
            for i, clause1 in enumerate(term_clauses):
                for clause2 in term_clauses[i+1:]:
                    if self._are_conflicting(clause1.text, clause2.text):
                        terms.append(UnfavorableTerm(
                            term="Conflicting Termination Clauses",
                            category="termination",
                            severity="high",
                            explanation=f"Clauses {clause1.reference} and {clause2.reference} contain conflicting termination terms, creating ambiguity.",
                            clause_reference=f"{clause1.reference}, {clause2.reference}",
                            suggested_fix="Clarify which termination provision takes precedence or consolidate into single consistent clause."
                        ))
                        break
        
        return terms
    
    def _are_conflicting(self, text1: str, text2: str) -> bool:
        """Check if two texts contain conflicting terms (simplified heuristic)"""
        # Extract numbers (notice periods, durations, etc.)
        nums1 = set(re.findall(r'\d+', text1))
        nums2 = set(re.findall(r'\d+', text2))
        
        # If they share the same type of clause but have different numbers, might be conflicting
        if nums1 and nums2 and len(nums1 & nums2) == 0:
            return True
        
        return False
    
    def _deduplicate_and_prioritize(self, terms: List[UnfavorableTerm]) -> List[UnfavorableTerm]:
        """Remove duplicates and sort by severity"""
        
        # Deduplicate by term + category
        seen = set()
        unique_terms = []
        
        for term in terms:
            key = (term.term, term.category)
            if key not in seen:
                seen.add(key)
                unique_terms.append(term)
        
        # Sort by severity (critical > high > medium)
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        unique_terms.sort(key=lambda t: severity_order.get(t.severity, 3))
        
        return unique_terms[:15]  # Top 15 unfavorable terms
    
    def get_severity_distribution(self, terms: List[UnfavorableTerm]) -> Dict[str, int]:
        """Get distribution of terms by severity"""
        distribution = {"critical": 0, "high": 0, "medium": 0}
        
        for term in terms:
            distribution[term.severity] = distribution.get(term.severity, 0) + 1
        
        log_info("Severity distribution", **distribution)
        
        return distribution
    
    def get_category_distribution(self, terms: List[UnfavorableTerm]) -> Dict[str, int]:
        """Get distribution of terms by category"""
        from collections import Counter
        
        categories = [t.category for t in terms]
        distribution = dict(Counter(categories))
        
        log_info("Category distribution", **distribution)
        
        return distribution