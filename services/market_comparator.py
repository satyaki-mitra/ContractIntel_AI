from sentence_transformers import util
import numpy as np

@dataclass
class MarketComparison:
    """Market comparison result"""
    clause_category: str
    user_clause: str
    market_standard: str
    similarity_score: float
    assessment: str  # "standard", "favorable", "unfavorable"
    explanation: str


class MarketComparator:
    """
    Compare contract terms to market standards
    Uses semantic similarity with embedding model
    """
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.embedding_model = None
        self._lazy_load()
        self._load_market_standards()
    
    def _lazy_load(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            print("[MarketComparator] Loading embedding model...")
            self.embedding_model = self.model_loader.load_embedding_model()
            print("[MarketComparator] Model loaded")
    
    def _load_market_standards(self):
        """Load market standard clause templates"""
        self.market_standards = {
            'non-compete': {
                'reasonable': "Employee agrees not to compete with Company in direct competitive business within 50 miles for 6 months after termination.",
                'standard': "Employee shall not engage in competitive activities with direct competitors for 12 months within the geographic area of Company operations.",
                'aggressive': "Employee shall not work in any capacity in the industry for 24 months globally."
            },
            'termination': {
                'reasonable': "Either party may terminate with 30 days written notice. Company shall pay severance equal to 2 months salary if terminated without cause.",
                'standard': "Either party may terminate with 30 days notice. Employee terminated without cause receives 1 month severance.",
                'aggressive': "Company may terminate immediately without cause or notice. Employee must provide 60 days notice."
            },
            'confidentiality': {
                'reasonable': "Confidential information remains confidential for 3 years after termination, limited to information marked confidential.",
                'standard': "Employee shall maintain confidentiality of proprietary information for 5 years after termination.",
                'aggressive': "All information learned during employment remains confidential perpetually."
            },
            'intellectual_property': {
                'reasonable': "Company owns work product created for Company during employment, excluding personal projects unrelated to Company business.",
                'standard': "All work product and inventions created during employment belong to Company.",
                'aggressive': "Company owns all intellectual property created by Employee during employment and for 12 months after, including personal projects."
            },
            'indemnification': {
                'reasonable': "Each party shall indemnify the other for losses arising from their respective negligence or willful misconduct, capped at fees paid.",
                'standard': "Employee shall indemnify Company for losses arising from Employee's breach or negligence.",
                'aggressive': "Employee shall indemnify Company for all claims, with unlimited liability including consequential damages."
            },
            'liability': {
                'reasonable': "Liability capped at 12 months of fees paid. No liability for indirect or consequential damages.",
                'standard': "Liability limited to direct damages only, capped at amount paid in preceding 12 months.",
                'aggressive': "No limitation on liability. Party liable for all damages including consequential, indirect, and punitive."
            }
        }
    
    def compare_to_market(self, clauses: List[ExtractedClause]) -> List[MarketComparison]:
        """Compare extracted clauses to market standards"""
        
        comparisons = []
        
        for clause in clauses:
            if clause.category in self.market_standards:
                comparison = self._compare_single_clause(clause)
                if comparison:
                    comparisons.append(comparison)
        
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
        
        # Calculate similarities
        similarities = util.cos_sim(user_embedding, standard_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_similarity = similarities[best_idx].item()
        best_standard_type = list(standards.keys())[best_idx]
        best_standard_text = standard_texts[best_idx]
        
        # Assess based on similarity and standard type
        assessment, explanation = self._assess_comparison(
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
            explanation=explanation
        )
    
    def _assess_comparison(self, standard_type: str, 
                          similarity: float,
                          category: str) -> Tuple[str, str]:
        """Assess if clause is favorable, standard, or unfavorable"""
        
        # High similarity to reasonable standard = favorable
        if standard_type == 'reasonable' and similarity >= 0.65:
            return "favorable", f"This {category} clause aligns with reasonable market standards (similarity: {similarity:.2f})"
        
        # High similarity to standard = acceptable
        elif standard_type == 'standard' and similarity >= 0.65:
            return "standard", f"This {category} clause matches typical market standards (similarity: {similarity:.2f})"
        
        # High similarity to aggressive = unfavorable
        elif standard_type == 'aggressive' and similarity >= 0.65:
            return "unfavorable", f"This {category} clause is more aggressive than market standards (similarity: {similarity:.2f})"
        
        # Moderate similarity
        elif 0.50 <= similarity < 0.65:
            if standard_type == 'reasonable':
                return "standard", f"This {category} clause is somewhat aligned with reasonable standards"
            elif standard_type == 'aggressive':
                return "concerning", f"This {category} clause shows some aggressive elements compared to market norms"
            else:
                return "standard", f"This {category} clause is within normal market range"
        
        # Low similarity - unclear
        else:
            return "unclear", f"This {category} clause is unique and difficult to compare to standard market terms"
    
    def get_recommendations(self, comparisons: List[MarketComparison]) -> List[str]:
        """Get actionable recommendations based on market comparison"""
        
        recommendations = []
        
        # Group by assessment
        unfavorable = [c for c in comparisons if c.assessment == "unfavorable"]
        concerning = [c for c in comparisons if c.assessment == "concerning"]
        
        # Priority recommendations for unfavorable terms
        for comp in unfavorable:
            recommendations.append(
                f"⚠️ {comp.clause_category.replace('_', ' ').title()}: "
                f"Negotiate to align with market standard: \"{comp.market_standard[:100]}...\""
            )
        
        # Secondary recommendations for concerning terms
        for comp in concerning:
            recommendations.append(
                f"📋 {comp.clause_category.replace('_', ' ').title()}: "
                f"Consider requesting modifications to match industry norms"
            )
        
        return recommendations[:5]  # Top 5 recommendations
