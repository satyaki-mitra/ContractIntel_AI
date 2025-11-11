"""
Negotiation Engine
Generates intelligent, prioritized negotiation talking points using LLM
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import sys
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_manager.llm_manager import LLMManager, LLMProvider
from services.clause_extractor import ExtractedClause
from services.term_analyzer import UnfavorableTerm
from services.protection_checker import MissingProtection
from services.risk_analyzer import RiskScore
from utils.logger import ContractAnalyzerLogger, log_info, log_error


@dataclass
class NegotiationPoint:
    """Single negotiation talking point"""
    priority: int  # 1=highest, 5=lowest
    category: str
    issue: str
    current_language: str
    proposed_language: str
    rationale: str
    fallback_position: Optional[str] = None
    estimated_difficulty: str = "medium"  # "easy", "medium", "hard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "priority": self.priority,
            "category": self.category,
            "issue": self.issue,
            "current_language": self.current_language,
            "proposed_language": self.proposed_language,
            "rationale": self.rationale,
            "fallback_position": self.fallback_position,
            "estimated_difficulty": self.estimated_difficulty
        }


class NegotiationEngine:
    """
    Generate intelligent negotiation strategy with LLM enhancement
    Combines rule-based prioritization with LLM-generated specific language
    """
    
    def __init__(self, llm_manager: LLMManager,
                 default_provider: LLMProvider = LLMProvider.OLLAMA):
        """
        Initialize negotiation engine
        
        Args:
            llm_manager: LLMManager instance
            default_provider: Default LLM provider
        """
        self.llm_manager = llm_manager
        self.default_provider = default_provider
        self.logger = ContractAnalyzerLogger.get_logger()
        
        log_info("NegotiationEngine initialized",
                default_provider=default_provider.value)
    
    @ContractAnalyzerLogger.log_execution_time("generate_negotiation_points")
    def generate_negotiation_points(self,
                                   risk_analysis: RiskScore,
                                   unfavorable_terms: List[UnfavorableTerm],
                                   missing_protections: List[MissingProtection],
                                   clauses: List[ExtractedClause],
                                   max_points: int = 7,
                                   provider: Optional[LLMProvider] = None) -> List[NegotiationPoint]:
        """
        Generate prioritized negotiation strategy
        
        Args:
            risk_analysis: Risk analysis results
            unfavorable_terms: Detected unfavorable terms
            missing_protections: Missing protections
            clauses: Extracted clauses
            max_points: Maximum negotiation points to generate
            provider: LLM provider
        
        Returns:
            List of NegotiationPoint objects sorted by priority
        """
        provider = provider or self.default_provider
        
        log_info("Starting negotiation points generation",
                max_points=max_points,
                unfavorable_terms=len(unfavorable_terms),
                missing_protections=len(missing_protections))
        
        negotiation_points = []
        
        # Priority 1: Critical unfavorable terms
        critical_terms = [t for t in unfavorable_terms if t.severity == "critical"]
        for term in critical_terms[:3]:
            point = self._create_point_from_term(term, clauses, priority=1)
            if point:
                negotiation_points.append(point)
        
        # Priority 2: Critical missing protections
        critical_protections = [p for p in missing_protections if p.importance == "critical"]
        for protection in critical_protections[:2]:
            point = self._create_point_from_protection(protection, priority=2)
            negotiation_points.append(point)
        
        # Priority 3: High unfavorable terms
        high_terms = [t for t in unfavorable_terms if t.severity == "high"]
        for term in high_terms[:2]:
            point = self._create_point_from_term(term, clauses, priority=3)
            if point:
                negotiation_points.append(point)
        
        # Priority 4: High-risk categories
        for category in risk_analysis.risk_factors[:2]:
            point = self._create_category_point(category, risk_analysis, clauses, priority=4)
            if point:
                negotiation_points.append(point)
        
        # Enhance with LLM-generated specific language
        enhanced_points = self._enhance_with_llm(negotiation_points[:max_points], provider)
        
        log_info(f"Negotiation points generation complete",
                total_points=len(enhanced_points))
        
        return enhanced_points[:max_points]
    
    def _create_point_from_term(self, term: UnfavorableTerm,
                               clauses: List[ExtractedClause],
                               priority: int) -> Optional[NegotiationPoint]:
        """Create negotiation point from unfavorable term"""
        
        # Find the actual clause
        clause = next((c for c in clauses if c.reference == term.clause_reference), None)
        if not clause:
            return None
        
        current = clause.text[:150] + "..." if len(clause.text) > 150 else clause.text
        
        # Use suggested fix from term or generate
        proposed = term.suggested_fix or self._generate_proposed_language(term)
        
        difficulty = "hard" if term.severity == "critical" else "medium"
        
        return NegotiationPoint(
            priority=priority,
            category=term.category,
            issue=term.term,
            current_language=current,
            proposed_language=proposed,
            rationale=term.explanation,
            fallback_position=self._generate_fallback(term),
            estimated_difficulty=difficulty
        )
    
    def _create_point_from_protection(self, protection: MissingProtection,
                                     priority: int) -> NegotiationPoint:
        """Create negotiation point from missing protection"""
        
        difficulty = "medium" if protection.importance == "critical" else "easy"
        
        return NegotiationPoint(
            priority=priority,
            category=protection.category,
            issue=f"Add {protection.protection}",
            current_language="[NOT PRESENT IN CONTRACT]",
            proposed_language=protection.recommendation,
            rationale=protection.explanation,
            fallback_position="If they refuse, document this gap and consider it in overall risk assessment",
            estimated_difficulty=difficulty
        )
    
    def _create_category_point(self, category: str,
                              risk_analysis: RiskScore,
                              clauses: List[ExtractedClause],
                              priority: int) -> Optional[NegotiationPoint]:
        """Create negotiation point for high-risk category"""
        
        category_clauses = [c for c in clauses if self._matches_category(c.category, category)]
        if not category_clauses:
            return None
        
        score = risk_analysis.category_scores.get(category, 0)
        
        return NegotiationPoint(
            priority=priority,
            category=category,
            issue=f"Reduce {category.replace('_', ' ')} risk (score: {score}/100)",
            current_language=f"Multiple clauses in this category present elevated risk",
            proposed_language=f"Request balanced terms for {category.replace('_', ' ')} provisions",
            rationale=f"This category scores {score}/100, indicating significant risk requiring mitigation",
            estimated_difficulty="medium"
        )
    
    def _generate_proposed_language(self, term: UnfavorableTerm) -> str:
        """Generate proposed language for unfavorable term"""
        
        proposals = {
            "unlimited_liability": "Add: 'Total liability capped at [amount] or 12 months fees paid, whichever is greater'",
            "sole_discretion": "Replace 'sole discretion' with 'reasonable discretion' or 'mutual agreement'",
            "at_will_termination": "Add: 'with [30-60] days notice and [X months] severance for termination without cause'",
            "wage_withholding": "Remove clause entirely - wage withholding is generally illegal",
            "non_compete_overly_broad": "Limit to: (a) 6-12 months, (b) direct competitors only, (c) reasonable geographic area"
        }
        
        term_key = term.term.lower().replace(" ", "_")
        return proposals.get(term_key, "[Request balanced, market-standard language]")
    
    def _generate_fallback(self, term: UnfavorableTerm) -> str:
        """Generate fallback negotiation position"""
        
        if term.severity == "critical":
            return "If they refuse removal/revision, request liability cap or consider walking away from deal"
        elif term.severity == "high":
            return "If they won't revise, request mutual application or shorter duration/narrower scope"
        else:
            return "If they won't budge, document concerns and assess if other favorable terms compensate"
    
    def _matches_category(self, clause_category: str, risk_category: str) -> bool:
        """Check if clause category matches risk category"""
        mapping = {
            "restrictive_covenants": ["non_compete", "confidentiality"],
            "termination_rights": ["termination"],
            "penalties_liability": ["indemnification", "liability"],
            "compensation_benefits": ["compensation"],
            "intellectual_property": ["intellectual_property"]
        }
        return clause_category in mapping.get(risk_category, [])
    
    def _enhance_with_llm(self, points: List[NegotiationPoint],
                         provider: LLMProvider) -> List[NegotiationPoint]:
        """Use LLM to enhance negotiation points with specific language"""
        
        if not points:
            return points
        
        log_info(f"Enhancing {len(points)} negotiation points with LLM")
        
        # Prepare context for LLM
        context = {
            "points": [
                {
                    "priority": p.priority,
                    "issue": p.issue,
                    "current": p.current_language[:200],
                    "category": p.category,
                    "rationale": p.rationale[:200]
                }
                for p in points
            ]
        }
        
        try:
            prompt = self._create_enhancement_prompt(context)
            
            response = self.llm_manager.complete(
                prompt=prompt,
                provider=provider,
                temperature=0.2,
                max_tokens=1500,
                fallback_providers=[LLMProvider.OPENAI],
                retry_on_error=True
            )
            
            if response.success:
                # Parse LLM response and enhance points
                enhanced = self._parse_llm_enhancements(response.text, points)
                log_info("LLM enhancement successful")
                return enhanced
            else:
                log_info("LLM enhancement failed, using original points")
                return points
        
        except Exception as e:
            log_error(e, context={"component": "NegotiationEngine", "operation": "enhance_with_llm"})
            return points  # Fallback to original
    
    def _create_enhancement_prompt(self, context: Dict) -> str:
        """Create prompt for LLM enhancement"""
        
        prompt = f"""You are an expert contract negotiation advisor. For each issue below, provide SPECIFIC proposed contract language that is:
1. Balanced and reasonable
2. Uses clear, professional legal language
3. Realistic for negotiation
4. Removes one-sided terms

Issues to address:
{json.dumps(context['points'], indent=2)}

For each issue, provide:
- Specific replacement clause language (not just suggestions)
- Rationale in business terms
- Fallback position if primary request is rejected

Format your response as clear, numbered points (1, 2, 3...) with "PROPOSED:", "RATIONALE:", and "FALLBACK:" sections for each.

Keep responses concise and actionable."""
        
        return prompt
    
    def _parse_llm_enhancements(self, llm_text: str, 
                               original_points: List[NegotiationPoint]) -> List[NegotiationPoint]:
        """Parse LLM response and apply enhancements to points"""
        
        # Simple parsing - look for PROPOSED: sections
        enhanced = []
        
        for i, point in enumerate(original_points):
            # Try to extract enhancement for this point
            pattern = rf"{i+1}[.\)]\s*.*?PROPOSED:\s*(.*?)(?:RATIONALE:|FALLBACK:|{i+2}\.|$)"
            match = re.search(pattern, llm_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                proposed = match.group(1).strip()
                if proposed and len(proposed) > 20:
                    point.proposed_language = proposed[:500]  # Limit length
            
            # Try to extract fallback
            fallback_pattern = rf"{i+1}[.\)]\s*.*?FALLBACK:\s*(.*?)(?:{i+2}\.|$)"
            fallback_match = re.search(fallback_pattern, llm_text, re.IGNORECASE | re.DOTALL)
            
            if fallback_match:
                fallback = fallback_match.group(1).strip()
                if fallback and len(fallback) > 10:
                    point.fallback_position = fallback[:300]
            
            enhanced.append(point)
        
        return enhanced
    
    def generate_negotiation_strategy_document(self,
                                              points: List[NegotiationPoint]) -> str:
        """
        Generate a formatted negotiation strategy document
        
        Returns:
            Formatted markdown document
        """
        
        doc = ["# Negotiation Strategy", "", "## Priority Ranking", ""]
        
        # Group by priority
        by_priority = {}
        for point in points:
            if point.priority not in by_priority:
                by_priority[point.priority] = []
            by_priority[point.priority].append(point)
        
        priority_labels = {
            1: "🔴 CRITICAL PRIORITY",
            2: "🟠 HIGH PRIORITY",
            3: "🟡 MEDIUM PRIORITY",
            4: "🟢 LOW PRIORITY"
        }
        
        for priority in sorted(by_priority.keys()):
            doc.append(f"### {priority_labels.get(priority, f'Priority {priority}')}")
            doc.append("")
            
            for point in by_priority[priority]:
                doc.append(f"#### {point.issue}")
                doc.append(f"**Category:** {point.category}")
                doc.append(f"**Difficulty:** {point.estimated_difficulty}")
                doc.append("")
                doc.append("**Current Language:**")
                doc.append(f"> {point.current_language}")
                doc.append("")
                doc.append("**Proposed Language:**")
                doc.append(f"{point.proposed_language}")
                doc.append("")
                doc.append("**Rationale:**")
                doc.append(f"{point.rationale}")
                doc.append("")
                if point.fallback_position:
                    doc.append("**Fallback Position:**")
                    doc.append(f"{point.fallback_position}")
                    doc.append("")
                doc.append("---")
                doc.append("")
        
        return "\n".join(doc)
    
    def get_critical_points(self, points: List[NegotiationPoint]) -> List[NegotiationPoint]:
        """Filter to only priority 1-2 points"""
        critical = [p for p in points if p.priority <= 2]
        log_info(f"Found {len(critical)} critical negotiation points")
        return critical
    
    def get_points_by_category(self, points: List[NegotiationPoint],
                              category: str) -> List[NegotiationPoint]:
        """Filter points by category"""
        filtered = [p for p in points if p.category == category]
        log_info(f"Found {len(filtered)} negotiation points in category '{category}'")
        return filtered

