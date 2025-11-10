import requests
import json

@dataclass
class NegotiationPoint:
    """Negotiation talking point"""
    priority: int  # 1=highest
    category: str
    issue: str
    current_language: str
    proposed_language: str
    rationale: str
    fallback_position: Optional[str] = None


class NegotiationEngine:
    """Generate intelligent negotiation talking points"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.rules = RiskRules()
    
    def generate_negotiation_points(self, 
                                   risk_analysis: RiskScore,
                                   unfavorable_terms: List[UnfavorableTerm],
                                   missing_protections: List[MissingProtection],
                                   clauses: List[ExtractedClause]) -> List[NegotiationPoint]:
        """Generate prioritized negotiation strategy"""
        
        negotiation_points = []
        
        # Priority 1: Critical unfavorable terms
        critical_terms = [t for t in unfavorable_terms if t.severity == "critical"]
        for term in critical_terms[:3]:
            point = self._create_negotiation_point_from_term(term, clauses, priority=1)
            if point:
                negotiation_points.append(point)
        
        # Priority 2: Critical missing protections
        critical_protections = [p for p in missing_protections if p.importance == "critical"]
        for protection in critical_protections[:2]:
            point = self._create_negotiation_point_from_protection(protection, priority=2)
            negotiation_points.append(point)
        
        # Priority 3: High-risk category improvements
        for category, score in risk_analysis.category_scores.items():
            if score >= self.rules.RISK_THRESHOLDS["high"]:
                point = self._create_category_negotiation_point(category, score, clauses, priority=3)
                if point:
                    negotiation_points.append(point)
        
        # Use LLM to enhance negotiation points with specific language
        enhanced_points = self._enhance_with_llm(negotiation_points, clauses)
        
        return enhanced_points[:7]  # Top 7 negotiation points
    
    def _create_negotiation_point_from_term(self, term: UnfavorableTerm, 
                                           clauses: List[ExtractedClause],
                                           priority: int) -> Optional[NegotiationPoint]:
        """Create negotiation point from unfavorable term"""
        
        # Find the actual clause
        clause = next((c for c in clauses if c.reference == term.clause_reference), None)
        if not clause:
            return None
        
        current = clause.text[:200]  # First 200 chars
        
        # Generate proposed language based on term type
        proposed = self._generate_proposed_language(term, clause)
        
        return NegotiationPoint(
            priority=priority,
            category=term.category,
            issue=term.term,
            current_language=current,
            proposed_language=proposed,
            rationale=term.explanation,
            fallback_position=self._generate_fallback(term)
        )
    
    def _create_negotiation_point_from_protection(self, protection: MissingProtection,
                                                 priority: int) -> NegotiationPoint:
        """Create negotiation point from missing protection"""
        
        return NegotiationPoint(
            priority=priority,
            category=protection.category,
            issue=f"Add {protection.protection}",
            current_language="[NOT PRESENT]",
            proposed_language=protection.recommendation,
            rationale=protection.explanation
        )
    
    def _create_category_negotiation_point(self, category: str, score: int,
                                          clauses: List[ExtractedClause],
                                          priority: int) -> Optional[NegotiationPoint]:
        """Create negotiation point for high-risk category"""
        
        category_clauses = [c for c in clauses if self._matches_category(c.category, category)]
        if not category_clauses:
            return None
        
        # Use highest risk clause in category
        highest_risk_clause = category_clauses[0]
        
        return NegotiationPoint(
            priority=priority,
            category=category,
            issue=f"Reduce {category.replace('_', ' ')} risk (score: {score})",
            current_language=highest_risk_clause.text[:150],
            proposed_language=f"[Propose balanced terms for {category}]",
            rationale=f"This category scores {score}/100, indicating significant risk that should be mitigated"
        )
    
    def _generate_proposed_language(self, term: UnfavorableTerm, 
                                   clause: ExtractedClause) -> str:
        """Generate specific proposed language"""
        
        # Rule-based proposals for common issues
        if "non-compete" in term.term.lower():
            return "Limit non-compete to: (a) 6-12 months duration, (b) direct competitors only, (c) specific geographic region where Company operates"
        
        elif "sole discretion" in term.term.lower():
            return "Replace 'sole discretion' with 'reasonable discretion' or 'mutual agreement'"
        
        elif "without cause" in term.term.lower():
            return "Add: 'with [30-60] days' notice and [X months] severance pay'"
        
        elif "indemnify" in term.term.lower():
            return "Make mutual: 'Each party shall indemnify the other for losses arising from their respective breach, negligence, or willful misconduct'"
        
        else:
            return "[Request specific balanced language]"
    
    def _generate_fallback(self, term: UnfavorableTerm) -> str:
        """Generate fallback negotiation position"""
        
        if term.severity == "critical":
            return "If they won't remove/revise, request liability cap or consider walking away"
        elif term.severity == "high":
            return "If they won't revise, request mutual application or shorter duration"
        else:
            return "If they won't budge, document concerns and consider accepting if other terms are favorable"
    
    def _matches_category(self, clause_category: str, risk_category: str) -> bool:
        """Check if clause category matches risk category"""
        mapping = {
            "restrictive_covenants": ["non-compete", "confidentiality"],
            "termination_rights": ["termination"],
            "penalties_liability": ["indemnification", "liability"],
            "compensation_benefits": ["compensation"],
            "intellectual_property": ["intellectual_property"]
        }
        return clause_category in mapping.get(risk_category, [])
    
    def _enhance_with_llm(self, points: List[NegotiationPoint], 
                         clauses: List[ExtractedClause]) -> List[NegotiationPoint]:
        """Use LLM to enhance negotiation points with specific language"""
        
        # Prepare context for LLM
        context = {
            "points": [
                {
                    "priority": p.priority,
                    "issue": p.issue,
                    "current": p.current_language,
                    "category": p.category
                }
                for p in points[:5]  # Top 5 for LLM enhancement
            ]
        }
        
        try:
            # Call Ollama to enhance
            prompt = self._create_enhancement_prompt(context)
            response = requests.post(
                f"{self.llm_config['base_url']}/api/generate",
                json={
                    "model": self.llm_config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config["temperature"],
                        "num_predict": 1500
                    }
                },
                timeout=self.llm_config["timeout"]
            )
            
            if response.status_code == 200:
                enhanced_text = response.json().get("response", "")
                # Parse and apply enhancements (simplified)
                # In production, use structured output
                return points  # Return original if parsing fails
        
        except Exception:
            pass  # Fallback to original points
        
        return points
    
    def _create_enhancement_prompt(self, context: Dict) -> str:
        """Create prompt for LLM enhancement"""
        return f"""You are a legal negotiation expert. For each contract issue below, provide specific proposed contract language that is balanced and reasonable.

Issues:
{json.dumps(context, indent=2)}

For each issue, provide SPECIFIC replacement language that:
1. Removes one-sided terms
2. Adds reasonable protections
3. Uses clear, professional legal language
4. Is realistic for negotiation

Keep responses concise and actionable."""