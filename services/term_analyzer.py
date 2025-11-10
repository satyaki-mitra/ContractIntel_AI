from typing import List, Dict
from dataclasses import dataclass
from typing import Optional
from config.risk_rules import RiskRules
from services.clause_extractor import ExtractedClause


@dataclass
class UnfavorableTerm:
    """Detected unfavorable term"""
    term: str
    category: str
    severity: str  # "critical", "high", "medium"
    explanation: str
    clause_reference: Optional[str] = None


class TermAnalyzer:
    """Detect unfavorable and one-sided terms"""
    
    def __init__(self):
        self.rules = RiskRules()
    
    def analyze_unfavorable_terms(self, contract_text: str, 
                                 clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """Detect all unfavorable terms in contract"""
        
        unfavorable_terms = []
        
        # Check each clause for unfavorable patterns
        for clause in clauses:
            terms = self._analyze_clause_terms(clause)
            unfavorable_terms.extend(terms)
        
        # Check full text for cross-clause issues
        cross_clause_terms = self._analyze_cross_clause_issues(contract_text, clauses)
        unfavorable_terms.extend(cross_clause_terms)
        
        # Deduplicate and prioritize
        return self._deduplicate_and_prioritize(unfavorable_terms)
    
    def _analyze_clause_terms(self, clause: ExtractedClause) -> List[UnfavorableTerm]:
        """Analyze a single clause for unfavorable terms"""
        terms = []
        text_lower = clause.text.lower()
        
        # Check against critical keywords
        for keyword, weight in self.rules.CRITICAL_KEYWORDS.items():
            if keyword in text_lower:
                terms.append(UnfavorableTerm(
                    term=f"Contains '{keyword}'",
                    category=clause.category,
                    severity="critical",
                    explanation=f"This clause contains '{keyword}', which is highly restrictive and may be unenforceable or overly punitive.",
                    clause_reference=clause.reference
                ))
        
        # Check for one-sided language
        one_sided_patterns = [
            (r'(employee|consultant|contractor)\s+shall.*but.*employer\s+may', 
             "One-sided obligations: You 'shall' but they 'may'"),
            (r'sole\s+discretion\s+of.*employer',
             "Employer has sole discretion (no mutual agreement)"),
            (r'without.*cause.*employer.*terminate',
             "Employer can terminate without cause (at-will provision)")
        ]
        
        for pattern, explanation in one_sided_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                terms.append(UnfavorableTerm(
                    term="One-sided provision",
                    category=clause.category,
                    severity="high",
                    explanation=explanation,
                    clause_reference=clause.reference
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
        
        return terms
    
    def _check_notice_imbalance(self, clauses: List[ExtractedClause]) -> Optional[UnfavorableTerm]:
        """Check if notice periods are imbalanced"""
        term_clauses = [c for c in clauses if c.category == "termination"]
        
        if not term_clauses:
            return None
        
        # Extract notice periods (simplified)
        text = " ".join([c.text for c in term_clauses])
        notice_pattern = r'(\d+)\s*days?\s*(notice|prior\s+notice)'
        matches = re.findall(notice_pattern, text, re.IGNORECASE)
        
        if len(matches) >= 2:
            periods = [int(m[0]) for m in matches]
            ratio = max(periods) / min(periods)
            
            if ratio >= 2:
                return UnfavorableTerm(
                    term="Imbalanced notice periods",
                    category="termination",
                    severity="high" if ratio >= 3 else "medium",
                    explanation=f"Notice period imbalance: {max(periods)} days vs {min(periods)} days. This creates an unfair burden.",
                    clause_reference=term_clauses[0].reference
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
            has_one_sided = any("you shall indemnify" in c.text.lower() for c in indem_clauses)
            has_mutual = any("mutual" in c.text.lower() or "both parties" in c.text.lower() for c in indem_clauses)
            
            if has_one_sided and not has_mutual:
                terms.append(UnfavorableTerm(
                    term="Non-reciprocal indemnification",
                    category="indemnification",
                    severity="critical",
                    explanation="You must indemnify them, but no reciprocal protection. This is one-sided liability exposure.",
                    clause_reference=indem_clauses[0].reference
                ))
        
        # Check IP assignment reciprocity
        ip_clauses = [c for c in clauses if c.category == "intellectual_property"]
        if ip_clauses:
            assigns_all = any("all" in c.text.lower() and "work product" in c.text.lower() for c in ip_clauses)
            excludes_prior = any("prior" in c.text.lower() or "existing" in c.text.lower() for c in ip_clauses)
            
            if assigns_all and not excludes_prior:
                terms.append(UnfavorableTerm(
                    term="Overly broad IP assignment",
                    category="intellectual_property",
                    severity="high",
                    explanation="Assigns all IP without excluding prior/personal work. Could claim ownership of your existing projects.",
                    clause_reference=ip_clauses[0].reference
                ))
        
        return terms
    
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
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        unique_terms.sort(key=lambda t: severity_order.get(t.severity, 3))
        
        return unique_terms[:10]  # Top 10 unfavorable terms

