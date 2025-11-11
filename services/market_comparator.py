"""
Market Comparator
Compares contract terms to market standards using semantic similarity
"""

import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.clause_extractor import ExtractedClause
from utils.logger import ContractAnalyzerLogger, log_info


@dataclass
class MarketComparison:
    """Market comparison result for a clause"""
    clause_category: str
    user_clause: str
    market_standard: str
    similarity_score: float
    assessment: str  # "standard", "favorable", "unfavorable", "aggressive"
    explanation: str
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "clause_category": self.clause_category,
            "user_clause": self.user_clause,
            "market_standard": self.market_standard,
            "similarity_score": round(self.similarity_score, 3),
            "assessment": self.assessment,
            "explanation": self.explanation,
            "recommendation": self.recommendation
        }


class MarketComparator:
    """
    Compare contract terms to market standards
    Uses semantic similarity with embedding model
    """
    
    def __init__(self, model_loader):
        """
        Initialize market comparator
        
        Args:
            model_loader: ModelLoader instance for embedding model
        """
        self.model_loader = model_loader
        self.embedding_model = None
        self.logger = ContractAnalyzerLogger.get_logger()
        
        self._lazy_load()
        self._load_market_standards()
        
        log_info("MarketComparator initialized")
    
    def _lazy_load(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            log_info("Loading embedding model for market comparison...")
            self.embedding_model = self.model_loader.load_embedding_model()
            log_info("Embedding model loaded")
    
    def _load_market_standards(self):
        """Load market standard clause templates"""
        self.market_standards = {
            'non_compete': {
                'reasonable': (
                    "Employee agrees not to compete with Company in direct competitive business "
                    "within 50 miles for 6 months after termination."
                ),
                'standard': (
                    "Employee shall not engage in competitive activities with direct competitors "
                    "for 12 months within the geographic area of Company operations."
                ),
                'aggressive': (
                    "Employee shall not work in any capacity in the industry for 24 months globally.")
            },
            'termination': {
                'reasonable': (
                    "Either party may terminate with 30 days written notice. Company shall pay "
                    "severance equal to 2 months salary if terminated without cause."
                ),
                'standard': (
                    "Either party may terminate with 30 days notice. Employee terminated without "
                    "cause receives 1 month severance."
                ),
                'aggressive': (
                    "Company may terminate immediately without cause or notice. Employee must "
                    "provide 60 days notice."
                )
            },
            'confidentiality': {
                'reasonable': (
                    "Confidential information remains confidential for 3 years after termination, "
                    "limited to information marked confidential."
                ),
                'standard': (
                    "Employee shall maintain confidentiality of proprietary information for 5 years "
                    "after termination."
                ),
                'aggressive': (
                    "All information learned during employment remains confidential perpetually."
                )
            },
            'intellectual_property': {
                'reasonable': (
                    "Company owns work product created for Company during employment, excluding "
                    "personal projects unrelated to Company business."
                ),
                'standard': (
                    "All work product and inventions created during employment belong to Company."
                ),
                'aggressive': (
                    "Company owns all intellectual property created by Employee during employment "
                    "and for 12 months after, including personal projects."
                )
            },
            'indemnification': {
                'reasonable': (
                    "Each party shall indemnify the other for losses arising from their respective "
                    "negligence or willful misconduct, capped at fees paid."
                ),
                'standard': (
                    "Employee shall indemnify Company for losses arising from Employee's breach "
                    "or negligence."
                ),
                'aggressive': (
                    "Employee shall indemnify Company for all claims, with unlimited liability "
                    "including consequential damages."
                )
            },
            'liability': {
                'reasonable': (
                    "Liability capped at 12 months of fees paid. No liability for indirect or "
                    "consequential damages."
                ),
                'standard': (
                    "Liability limited to direct damages only, capped at amount paid in preceding "
                    "12 months."
                ),
                'aggressive': (
                    "No limitation on liability. Party liable for all damages including consequential, "
                    "indirect, and punitive."
                )
            },
            'compensation': {
                'reasonable': (
                    "Base salary of $X per year, paid bi-weekly. Bonus of up to Y% based on clear "
                    "performance metrics. Annual review guaranteed."
                ),
                'standard': (
                    "Annual salary of $X payable per company payroll schedule. Discretionary bonus "
                    "may be awarded."
                ),
                'aggressive': (
                    "Compensation to be determined. Subject to review and modification at company's "
                    "sole discretion."
                )
            },
            'warranty': {
                'reasonable': (
                    "Services performed in professional manner consistent with industry standards. "
                    "Limited warranty for 90 days."
                ),
                'standard': (
                    "Work performed in good faith. Warranty disclaimers for merchantability and "
                    "fitness for purpose."
                ),
                'aggressive': (
                    "All warranties disclaimed. No guarantee of results. AS-IS with no recourse."
                )
            }
        }
        
        log_info(f"Loaded market standards for {len(self.market_standards)} categories")
    
    @ContractAnalyzerLogger.log_execution_time("compare_to_market")
    def compare_to_market(self, clauses: List[ExtractedClause]) -> List[MarketComparison]:
        """
        Compare extracted clauses to market standards
        
        Args:
            clauses: List of extracted clauses
        
        Returns:
            List of MarketComparison objects
        """
        
        log_info(f"Starting market comparison for {len(clauses)} clauses")
        
        comparisons = []
        
        for clause in clauses:
            if clause.category in self.market_standards:
                comparison = self._compare_single_clause(clause)
                if comparison:
                    comparisons.append(comparison)
        
        log_info(f"Market comparison complete",
                total_comparisons=len(comparisons),
                unfavorable=sum(1 for c in comparisons if c.assessment == "unfavorable"))
        
        return comparisons
    
    def _compare_single_clause(self, clause: ExtractedClause) -> Optional[MarketComparison]:
        """Compare single clause to market standards"""
        
        # Get market standards for this category
        standards = self.market_standards.get(clause.category)
        if not standards:
            return None
        
        # Encode user clause
        user_embedding = self.embedding_model.encode(clause.text, convert_to_tensor=True)
        
        # Encode market standards
        standard_texts = list(standards.values())
        standard_embeddings = self.embedding_model.encode(standard_texts, convert_to_tensor=True)
        
        # Calculate similarities using cosine similarity
        similarities = []
        for std_emb in standard_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                user_embedding.unsqueeze(0),
                std_emb.unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # Find best match
        best_idx = similarities.index(max(similarities))
        best_similarity = similarities[best_idx]
        best_standard_type = list(standards.keys())[best_idx]
        best_standard_text = standard_texts[best_idx]
        
        # Assess based on similarity and standard type
        assessment, explanation, recommendation = self._assess_comparison(
            best_standard_type,
            best_similarity,
            clause.category
        )
        
        return MarketComparison(
            clause_category=clause.category,
            user_clause=clause.text[:150] + "..." if len(clause.text) > 150 else clause.text,
            market_standard=best_standard_text,
            similarity_score=best_similarity,
            assessment=assessment,
            explanation=explanation,
            recommendation=recommendation
        )
    
    def _assess_comparison(self, standard_type: str,
                          similarity: float,
                          category: str) -> Tuple[str, str, str]:
        """
        Assess if clause is favorable, standard, or unfavorable
        
        Returns:
            (assessment, explanation, recommendation) tuple
        """
        
        # High similarity to reasonable standard = favorable
        if standard_type == 'reasonable' and similarity >= 0.65:
            return (
                "favorable",
                f"This {category} clause aligns with reasonable market standards (similarity: {similarity:.2%})",
                "This is a fair term. Consider accepting as-is or requesting minor improvements."
            )
        
        # High similarity to standard = acceptable
        elif standard_type == 'standard' and similarity >= 0.65:
            return (
                "standard",
                f"This {category} clause matches typical market standards (similarity: {similarity:.2%})",
                "This is standard market practice. Acceptable but could negotiate for better terms."
            )
        
        # High similarity to aggressive = unfavorable
        elif standard_type == 'aggressive' and similarity >= 0.65:
            return (
                "unfavorable",
                f"This {category} clause is more aggressive than market standards (similarity: {similarity:.2%})",
                "This is unfavorable. Strongly recommend negotiating to align with market norms."
            )
        
        # Moderate similarity
        elif 0.50 <= similarity < 0.65:
            if standard_type == 'reasonable':
                return (
                    "standard",
                    f"This {category} clause is somewhat aligned with reasonable standards",
                    "Consider requesting adjustments to better align with favorable market terms."
                )
            elif standard_type == 'aggressive':
                return (
                    "concerning",
                    f"This {category} clause shows some aggressive elements compared to market norms",
                    "Recommend negotiation to remove unfavorable provisions."
                )
            else:
                return (
                    "standard",
                    f"This {category} clause is within normal market range",
                    "Review carefully but likely acceptable if other terms are favorable."
                )
        
        # Low similarity - unclear
        else:
            return (
                "unclear",
                f"This {category} clause is unique and difficult to compare to standard market terms",
                "Seek legal counsel for specialized assessment of these non-standard terms."
            )
    
    def get_recommendations(self, comparisons: List[MarketComparison]) -> List[str]:
        """
        Get actionable recommendations based on market comparison
        
        Args:
            comparisons: List of market comparisons
        
        Returns:
            List of recommendation strings
        """
        
        recommendations = []
        
        # Group by assessment
        unfavorable = [c for c in comparisons if c.assessment == "unfavorable"]
        concerning = [c for c in comparisons if c.assessment == "concerning"]
        
        # Priority recommendations for unfavorable terms
        for comp in unfavorable[:5]:  # Top 5
            recommendations.append(
                f"⚠️ {comp.clause_category.replace('_', ' ').title()}: "
                f"Negotiate to align with market standard. {comp.recommendation}"
            )
        
        # Secondary recommendations for concerning terms
        for comp in concerning[:3]:  # Top 3
            recommendations.append(
                f"📋 {comp.clause_category.replace('_', ' ').title()}: "
                f"Consider requesting modifications. {comp.recommendation}"
            )
        
        log_info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def get_unfavorable_comparisons(self, comparisons: List[MarketComparison]) -> List[MarketComparison]:
        """Filter to only unfavorable comparisons"""
        unfavorable = [c for c in comparisons if c.assessment in ["unfavorable", "aggressive"]]
        
        log_info(f"Found {len(unfavorable)} unfavorable market comparisons")
        
        return unfavorable
    
    def get_favorable_comparisons(self, comparisons: List[MarketComparison]) -> List[MarketComparison]:
        """Filter to only favorable comparisons"""
        favorable = [c for c in comparisons if c.assessment == "favorable"]
        
        log_info(f"Found {len(favorable)} favorable market comparisons")
        
        return favorable
    
    def get_comparison_summary(self, comparisons: List[MarketComparison]) -> Dict[str, Any]:
        """
        Get summary statistics of market comparisons
        
        Returns:
            Dictionary with summary statistics
        """
        
        if not comparisons:
            return {
                "total": 0,
                "favorable": 0,
                "standard": 0,
                "unfavorable": 0,
                "concerning": 0,
                "unclear": 0,
                "avg_similarity": 0.0
            }
        
        assessments = [c.assessment for c in comparisons]
        similarities = [c.similarity_score for c in comparisons]
        
        summary = {
            "total": len(comparisons),
            "favorable": assessments.count("favorable"),
            "standard": assessments.count("standard"),
            "unfavorable": assessments.count("unfavorable"),
            "concerning": assessments.count("concerning"),
            "unclear": assessments.count("unclear"),
            "avg_similarity": round(sum(similarities) / len(similarities), 3)
        }
        
        log_info("Comparison summary", **summary)
        
        return summary
    
    def compare_specific_text(self, text: str, 
                             category: str) -> Optional[MarketComparison]:
        """
        Compare specific text to market standards
        
        Args:
            text: Clause text to compare
            category: Category of the clause
        
        Returns:
            MarketComparison object or None
        """
        
        # Create temporary ExtractedClause
        temp_clause = ExtractedClause(
            text=text,
            reference="Manual",
            category=category,
            confidence=1.0,
            start_pos=0,
            end_pos=len(text),
            extraction_method="manual",
            risk_indicators=[],
            legal_bert_score=0.0
        )
        
        return self._compare_single_clause(temp_clause)
    
    def get_best_practice_example(self, category: str) -> Optional[str]:
        """
        Get best practice example for a category
        
        Args:
            category: Clause category
        
        Returns:
            Best practice example text or None
        """
        
        standards = self.market_standards.get(category)
        if standards and 'reasonable' in standards:
            return standards['reasonable']
        
        return None
    
    def get_market_range(self, category: str) -> Optional[Dict[str, str]]:
        """
        Get the full market range for a category
        
        Args:
            category: Clause category
        
        Returns:
            Dictionary with reasonable/standard/aggressive examples
        """
        
        return self.market_standards.get(category)