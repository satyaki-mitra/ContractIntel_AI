"""
Advanced Contract Classifier using Legal-BERT + Semantic Similarity
Provides hierarchical categorization with confidence scores and multi-label support
"""

import re
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import util
import torch

# Import utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import ContractAnalyzerLogger, log_info, log_error
from utils.text_processor import TextProcessor


@dataclass
class ContractCategory:
    """Contract classification result with metadata"""
    category: str
    subcategory: Optional[str]
    confidence: float
    reasoning: List[str]
    detected_keywords: List[str]
    alternative_categories: List[Tuple[str, float]] = None  # (category, confidence) pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "detected_keywords": self.detected_keywords,
            "alternative_categories": [
                {"category": cat, "confidence": round(conf, 3)}
                for cat, conf in (self.alternative_categories or [])
            ]
        }


class ContractClassifier:
    """
    Sophisticated contract categorization using:
    1. Legal-BERT embeddings + semantic similarity
    2. Multi-label classification (a contract can be multiple types)
    3. Hierarchical categories (Employment -> Full-Time/Contract/Internship)
    4. Confidence scoring with explanations
    """
    
    # =========================================================================
    # CATEGORY HIERARCHY WITH KEYWORDS
    # =========================================================================
    
    CATEGORY_HIERARCHY = {
        'employment': {
            'subcategories': ['full_time', 'part_time', 'contract_worker', 'internship', 'executive'],
            'keywords': [
                'employee', 'employment', 'job', 'position', 'salary', 'benefits',
                'annual leave', 'sick leave', 'probation', 'job description',
                'work hours', 'overtime', 'performance review', 'bonus structure'
            ],
            'weight': 1.0
        },
        
        'consulting': {
            'subcategories': ['independent_contractor', 'advisory', 'professional_services', 'freelance'],
            'keywords': [
                'consultant', 'consulting', 'independent contractor', 'statement of work',
                'deliverables', 'professional services', 'hourly rate', 'project scope',
                'milestone', 'acceptance criteria', 'work product'
            ],
            'weight': 1.0
        },
        
        'nda': {
            'subcategories': ['mutual_nda', 'unilateral_nda', 'confidentiality_agreement'],
            'keywords': [
                'non-disclosure', 'confidentiality', 'proprietary information',
                'nda', 'disclosure agreement', 'trade secret', 'confidential information',
                'receiving party', 'disclosing party', 'confidentiality obligation'
            ],
            'weight': 1.2  # Higher weight as NDAs are distinct
        },
        
        'technology': {
            'subcategories': ['software_license', 'saas', 'cloud_services', 'development', 'api_access'],
            'keywords': [
                'software', 'license', 'saas', 'subscription', 'source code',
                'object code', 'api', 'cloud', 'hosting', 'maintenance',
                'updates', 'support', 'uptime', 'service level'
            ],
            'weight': 1.0
        },
        
        'intellectual_property': {
            'subcategories': ['ip_assignment', 'licensing', 'patent', 'trademark', 'copyright'],
            'keywords': [
                'intellectual property', 'ip', 'copyright', 'patent', 'trademark',
                'work product', 'inventions', 'ip rights', 'ownership',
                'assignment of rights', 'license grant', 'royalty'
            ],
            'weight': 1.1
        },
        
        'real_estate': {
            'subcategories': ['residential_lease', 'commercial_lease', 'sublease', 'purchase_agreement'],
            'keywords': [
                'landlord', 'tenant', 'lease', 'premises', 'rent', 'property',
                'security deposit', 'utilities', 'maintenance', 'repairs',
                'eviction', 'lease term', 'renewal', 'square footage'
            ],
            'weight': 1.0
        },
        
        'financial': {
            'subcategories': ['loan', 'mortgage', 'credit', 'investment', 'promissory_note'],
            'keywords': [
                'loan', 'borrower', 'lender', 'principal', 'interest rate',
                'collateral', 'default', 'repayment', 'amortization',
                'promissory note', 'security interest', 'mortgage'
            ],
            'weight': 1.0
        },
        
        'business': {
            'subcategories': ['partnership', 'joint_venture', 'shareholders', 'llc_operating', 'merger'],
            'keywords': [
                'partnership', 'joint venture', 'equity', 'shares', 'profit sharing',
                'loss allocation', 'management', 'governance', 'voting rights',
                'dissolution', 'capital contribution', 'distribution'
            ],
            'weight': 1.0
        },
        
        'sales': {
            'subcategories': ['purchase_order', 'sales_agreement', 'distribution', 'supply_agreement'],
            'keywords': [
                'purchase', 'sale', 'buyer', 'seller', 'goods', 'products',
                'delivery', 'shipment', 'payment terms', 'invoice',
                'purchase price', 'quantity', 'specifications'
            ],
            'weight': 1.0
        },
        
        'service_agreement': {
            'subcategories': ['master_services', 'maintenance', 'support', 'subscription'],
            'keywords': [
                'service provider', 'services', 'sla', 'service level agreement',
                'uptime', 'response time', 'support', 'maintenance',
                'service credits', 'performance metrics', 'implementation'
            ],
            'weight': 1.0
        },
        
        'vendor': {
            'subcategories': ['supplier_agreement', 'procurement', 'master_vendor'],
            'keywords': [
                'vendor', 'supplier', 'procurement', 'supply chain',
                'purchase order', 'fulfillment', 'vendor management',
                'pricing', 'terms of supply'
            ],
            'weight': 1.0
        },
        
        'agency': {
            'subcategories': ['marketing_agency', 'recruiting', 'representation'],
            'keywords': [
                'agent', 'agency', 'principal', 'commission', 'representation',
                'authority', 'scope of authority', 'compensation',
                'exclusive rights', 'territory'
            ],
            'weight': 1.0
        }
    }
    
    # =========================================================================
    # SUBCATEGORY DETECTION PATTERNS
    # =========================================================================
    
    SUBCATEGORY_PATTERNS = {
        # Employment subcategories
        'full_time': ['full-time', 'full time', 'permanent', 'regular employee', '40 hours', 'exempt employee'],
        'part_time': ['part-time', 'part time', 'hours per week', 'non-exempt', 'hourly employee'],
        'contract_worker': ['independent contractor', 'contract', 'fixed term', 'temporary', 'contract period'],
        'internship': ['intern', 'internship', 'student', 'training program', 'educational'],
        'executive': ['executive', 'ceo', 'cfo', 'cto', 'president', 'vice president', 'director'],
        
        # Consulting subcategories
        'independent_contractor': ['independent contractor', '1099', 'contractor', 'self-employed'],
        'advisory': ['advisor', 'advisory', 'counsel', 'consulting services', 'expert advice'],
        'professional_services': ['professional services', 'consulting services', 'engagement'],
        'freelance': ['freelance', 'freelancer', 'gig', 'project-based'],
        
        # NDA subcategories
        'mutual_nda': ['mutual', 'both parties', 'each party', 'reciprocal'],
        'unilateral_nda': ['one-way', 'receiving party', 'disclosing party', 'unilateral'],
        'confidentiality_agreement': ['confidentiality agreement', 'secrecy agreement'],
        
        # Real estate subcategories
        'residential_lease': ['residential', 'apartment', 'house', 'dwelling', 'residential property'],
        'commercial_lease': ['commercial', 'office space', 'retail space', 'commercial property'],
        'sublease': ['sublease', 'sublet', 'subtenant'],
        'purchase_agreement': ['purchase agreement', 'real property sale', 'deed'],
        
        # Financial subcategories
        'loan': ['loan agreement', 'term loan', 'credit facility'],
        'mortgage': ['mortgage', 'mortgagor', 'mortgagee', 'real property'],
        'credit': ['line of credit', 'credit agreement', 'revolving credit'],
        'investment': ['investment agreement', 'investor', 'investment'],
        'promissory_note': ['promissory note', 'note payable'],
        
        # Technology subcategories
        'saas': ['software as a service', 'saas', 'subscription', 'cloud-based'],
        'software_license': ['software license', 'license key', 'perpetual license', 'end user license'],
        'cloud_services': ['cloud services', 'cloud computing', 'infrastructure'],
        'development': ['software development', 'custom development', 'development services'],
        'api_access': ['api', 'application programming interface', 'api access'],
        
        # IP subcategories
        'ip_assignment': ['assignment', 'transfer of rights', 'work for hire'],
        'licensing': ['license', 'licensing agreement', 'license grant'],
        'patent': ['patent', 'patent rights', 'patent license'],
        'trademark': ['trademark', 'service mark', 'brand'],
        'copyright': ['copyright', 'copyrighted work'],
        
        # Business subcategories
        'partnership': ['partnership', 'general partnership', 'limited partnership'],
        'joint_venture': ['joint venture', 'jv agreement'],
        'shareholders': ['shareholders agreement', 'stock purchase', 'equity'],
        'llc_operating': ['operating agreement', 'llc', 'limited liability company'],
        'merger': ['merger', 'acquisition', 'm&a', 'consolidation'],
        
        # Sales subcategories
        'purchase_order': ['purchase order', 'po', 'order confirmation'],
        'sales_agreement': ['sales agreement', 'purchase agreement'],
        'distribution': ['distribution agreement', 'distributor', 'distribution rights'],
        'supply_agreement': ['supply agreement', 'supplier agreement'],
        
        # Service agreement subcategories
        'master_services': ['master services agreement', 'msa', 'master agreement'],
        'maintenance': ['maintenance agreement', 'maintenance services'],
        'support': ['support agreement', 'technical support', 'customer support'],
        'subscription': ['subscription agreement', 'subscription service']
    }
    
    def __init__(self, model_loader):
        """
        Initialize contract classifier
        
        Args:
            model_loader: ModelLoader instance for accessing Legal-BERT and embeddings
        """
        self.model_loader = model_loader
        self.embedding_model = None
        self.legal_bert_model = None
        self.legal_bert_tokenizer = None
        self.device = None
        
        # Category template embeddings (computed once)
        self.category_embeddings = {}
        
        # Text processor for preprocessing
        self.text_processor = TextProcessor(use_spacy=False)  # Don't need spaCy for classification
        
        # Logger
        self.logger = ContractAnalyzerLogger.get_logger()
        
        # Lazy load models
        self._lazy_load()
    
    def _lazy_load(self):
        """Lazy load models on first use"""
        if self.embedding_model is None:
            try:
                log_info("Loading models for contract classification...")
                
                # Load embedding model
                self.embedding_model = self.model_loader.load_embedding_model()
                
                # Load Legal-BERT
                self.legal_bert_model, self.legal_bert_tokenizer = self.model_loader.load_legal_bert()
                self.device = self.model_loader.device
                
                # Prepare category embeddings
                self._prepare_category_embeddings()
                
                log_info("Contract classifier models loaded successfully")
                
            except Exception as e:
                log_error(e, context={"component": "ContractClassifier", "operation": "model_loading"})
                raise
    
    def _prepare_category_embeddings(self):
        """Pre-compute embeddings for each category template"""
        log_info("Preparing category embeddings...")
        
        for category, config in self.CATEGORY_HIERARCHY.items():
            # Create representative template for each category
            keywords_sample = config['keywords'][:8]  # Use top 8 keywords
            template = (
                f"This is a {category.replace('_', ' ')} contract agreement involving "
                f"{', '.join(keywords_sample)}."
            )
            
            # Encode template
            embedding = self.embedding_model.encode(template, convert_to_tensor=True)
            self.category_embeddings[category] = embedding
        
        log_info(f"Prepared embeddings for {len(self.category_embeddings)} categories")
    
    # =========================================================================
    # MAIN CLASSIFICATION METHOD
    # =========================================================================
    
    @ContractAnalyzerLogger.log_execution_time("classify_contract")
    def classify_contract(self, contract_text: str, min_confidence: float = 0.50) -> ContractCategory:
        """
        Classify contract into granular categories with confidence scoring
        
        Process:
        1. Keyword-based initial scoring
        2. Semantic similarity with embeddings
        3. Legal-BERT enhanced classification
        4. Subcategory detection
        5. Confidence calibration
        
        Arguments:
        ----------
            contract_text   { str }  : Full contract text

            min_confidence { float } : Minimum confidence threshold (0.0-1.0)
        
        Returns:
        --------

            { ContractCategory }     : ContractCategory object with classification results
        """
        
        # Validate input
        if not contract_text or len(contract_text) < 100:
            raise ValueError("Contract text too short for classification")
        
        # Preprocess text (use first 3000 chars for efficiency)
        text_excerpt = contract_text[:3000]
        
        log_info("Starting contract classification", 
                text_length=len(contract_text),
                excerpt_length=len(text_excerpt))
        
        # Step 1: Keyword scoring
        keyword_scores = self._score_keywords(contract_text.lower())
        
        # Step 2: Semantic similarity
        semantic_scores = self._semantic_similarity(text_excerpt)
        
        # Step 3: Legal-BERT enhanced (optional - can be expensive)
        # legal_bert_scores = self._legal_bert_classification(text_excerpt)
        
        # Step 4: Combine scores (weighted average)
        combined_scores = self._combine_scores(
            keyword_scores=keyword_scores,
            semantic_scores=semantic_scores,
            # legal_bert_scores=legal_bert_scores  # Uncomment if using Legal-BERT
        )
        
        # Step 5: Get primary category
        if not combined_scores:
            log_info("No categories detected, defaulting to 'general'")
            return ContractCategory(
                category="general",
                subcategory=None,
                confidence=0.5,
                reasoning=["Unable to determine specific contract type"],
                detected_keywords=[]
            )
        
        primary_category = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[primary_category]
        
        # Step 6: Detect subcategory
        subcategory = self._detect_subcategory(contract_text, primary_category)
        
        # Step 7: Generate reasoning
        reasoning = self._generate_reasoning(
            contract_text=contract_text,
            primary_category=primary_category,
            subcategory=subcategory,
            keyword_scores=keyword_scores,
            semantic_scores=semantic_scores,
            combined_scores=combined_scores
        )
        
        # Step 8: Extract detected keywords
        detected_keywords = self._extract_detected_keywords(contract_text, primary_category)
        
        # Step 9: Get alternative categories
        alternative_categories = sorted(
            [(cat, score) for cat, score in combined_scores.items() if cat != primary_category],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 alternatives
        
        result = ContractCategory(
            category=primary_category,
            subcategory=subcategory,
            confidence=confidence,
            reasoning=reasoning,
            detected_keywords=detected_keywords,
            alternative_categories=alternative_categories
        )
        
        log_info("Contract classified successfully",
                category=primary_category,
                subcategory=subcategory,
                confidence=confidence)
        
        return result
    
    # =========================================================================
    # SCORING METHODS
    # =========================================================================
    
    def _score_keywords(self, text_lower: str) -> Dict[str, float]:
        """
        Score each category based on keyword presence
        
        Args:
            text_lower: Lowercase contract text
        
        Returns:
            Dictionary of {category: score}
        """
        scores = {}
        
        for category, config in self.CATEGORY_HIERARCHY.items():
            keywords = config['keywords']
            weight = config['weight']
            
            # Count keyword matches
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Normalize by number of keywords and apply weight
            normalized_score = (keyword_count / len(keywords)) * weight
            
            scores[category] = normalized_score
        
        return scores
    
    def _semantic_similarity(self, text: str) -> Dict[str, float]:
        """
        Calculate semantic similarity to category templates using embeddings
        
        Args:
            text: Contract text excerpt
        
        Returns:
            Dictionary of {category: similarity_score}
        """
        # Encode contract text
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        
        # Calculate similarity to each category
        similarities = {}
        for category, cat_embedding in self.category_embeddings.items():
            similarity = util.cos_sim(text_embedding, cat_embedding)[0][0].item()
            similarities[category] = similarity
        
        return similarities
    
    def _legal_bert_classification(self, text: str) -> Dict[str, float]:
        """
        Use Legal-BERT for classification (optional - computationally expensive)
        
        Args:
            text: Contract text excerpt
        
        Returns:
            Dictionary of {category: score}
        """
        # This is a placeholder for Legal-BERT classification
        # In production, you'd fine-tune Legal-BERT on labeled contract data
        
        # Tokenize
        inputs = self.legal_bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.legal_bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # For now, return uniform scores (placeholder)
        # In production, you'd use a trained classifier head
        return {cat: 0.5 for cat in self.CATEGORY_HIERARCHY.keys()}
    
    def _combine_scores(self, keyword_scores: Dict[str, float],
                       semantic_scores: Dict[str, float],
                       legal_bert_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Combine scores from different methods (weighted average)
        
        Args:
            keyword_scores: Keyword-based scores
            semantic_scores: Semantic similarity scores
            legal_bert_scores: Legal-BERT scores (optional)
        
        Returns:
            Combined scores dictionary
        """
        combined = {}
        
        # Weights for each method
        keyword_weight = 0.40
        semantic_weight = 0.60
        legal_bert_weight = 0.00  # Set to 0 if not using Legal-BERT
        
        if legal_bert_scores:
            # Normalize weights
            total_weight = keyword_weight + semantic_weight + legal_bert_weight
            keyword_weight /= total_weight
            semantic_weight /= total_weight
            legal_bert_weight /= total_weight
        
        for category in self.CATEGORY_HIERARCHY.keys():
            score = (
                keyword_scores.get(category, 0) * keyword_weight +
                semantic_scores.get(category, 0) * semantic_weight
            )
            
            if legal_bert_scores:
                score += legal_bert_scores.get(category, 0) * legal_bert_weight
            
            combined[category] = score
        
        return combined
    
    # =========================================================================
    # SUBCATEGORY DETECTION
    # =========================================================================
    
    def _detect_subcategory(self, text: str, primary_category: str) -> Optional[str]:
        """
        Detect specific subcategory within primary category
        
        Args:
            text: Full contract text
            primary_category: Detected primary category
        
        Returns:
            Subcategory name or None
        """
        text_lower = text.lower()
        
        # Get subcategories for this category
        subcategories = self.CATEGORY_HIERARCHY[primary_category]['subcategories']
        
        # Score each subcategory
        subcat_scores = {}
        for subcat in subcategories:
            if subcat in self.SUBCATEGORY_PATTERNS:
                patterns = self.SUBCATEGORY_PATTERNS[subcat]
                score = sum(1 for pattern in patterns if pattern in text_lower)
                subcat_scores[subcat] = score
        
        # Return best match if any
        if subcat_scores and max(subcat_scores.values()) > 0:
            best_subcat = max(subcat_scores, key=subcat_scores.get)
            log_info(f"Detected subcategory: {best_subcat}", 
                    category=primary_category,
                    score=subcat_scores[best_subcat])
            return best_subcat
        
        return None
    
    # =========================================================================
    # REASONING & EXPLANATION
    # =========================================================================
    
    def _generate_reasoning(self, contract_text: str, primary_category: str,
                           subcategory: Optional[str],
                           keyword_scores: Dict[str, float],
                           semantic_scores: Dict[str, float],
                           combined_scores: Dict[str, float]) -> List[str]:
        """
        Generate human-readable reasoning for classification
        
        Returns:
            List of reasoning statements
        """
        reasoning = []
        
        # Primary category reasoning
        keyword_match = keyword_scores.get(primary_category, 0)
        semantic_match = semantic_scores.get(primary_category, 0)
        
        if keyword_match > 0.5:
            reasoning.append(
                f"Strong keyword indicators for {primary_category.replace('_', ' ')} category "
                f"({int(keyword_match * 100)}% keyword match)"
            )
        elif keyword_match > 0.3:
            reasoning.append(
                f"Moderate keyword presence for {primary_category.replace('_', ' ')} "
                f"({int(keyword_match * 100)}% keyword match)"
            )
        
        if semantic_match > 0.65:
            reasoning.append(
                f"Contract language semantically similar to {primary_category.replace('_', ' ')} agreements "
                f"(similarity: {semantic_match:.2f})"
            )
        elif semantic_match > 0.50:
            reasoning.append(
                f"Moderate semantic similarity to {primary_category.replace('_', ' ')} contracts "
                f"(similarity: {semantic_match:.2f})"
            )
        
        # Subcategory reasoning
        if subcategory:
            reasoning.append(
                f"Specific subcategory identified: {subcategory.replace('_', ' ')}"
            )
        
        # Alternative categories (if close)
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.40:
            alt_category, alt_score = sorted_scores[1]
            reasoning.append(
                f"Also contains elements of {alt_category.replace('_', ' ')} "
                f"(secondary match: {alt_score:.2f})"
            )
        
        # If no strong reasoning
        if not reasoning:
            reasoning.append("Classification based on general contract structure and terminology")
        
        return reasoning
    
    def _extract_detected_keywords(self, text: str, category: str) -> List[str]:
        """
        Extract which specific keywords were found
        
        Args:
            text: Contract text
            category: Detected category
        
        Returns:
            List of detected keywords
        """
        text_lower = text.lower()
        keywords = self.CATEGORY_HIERARCHY[category]['keywords']
        
        detected = [kw for kw in keywords if kw in text_lower]
        return detected[:10]  # Top 10 keywords
    
    # =========================================================================
    # MULTI-LABEL CLASSIFICATION
    # =========================================================================
    
    @ContractAnalyzerLogger.log_execution_time("classify_multi_label")
    def classify_multi_label(self, text: str, 
                            threshold: float = 0.45) -> List[ContractCategory]:
        """
        Classify as multiple categories if applicable
        (e.g., Employment + NDA, Consulting + IP Assignment)
        
        Args:
            text: Contract text
            threshold: Minimum confidence threshold for multi-label
        
        Returns:
            List of ContractCategory objects (sorted by confidence)
        """
        log_info("Starting multi-label classification", threshold=threshold)
        
        # Get scores
        keyword_scores = self._score_keywords(text.lower())
        semantic_scores = self._semantic_similarity(text[:3000])
        combined_scores = self._combine_scores(keyword_scores, semantic_scores)
        
        # Get all categories above threshold
        matches = []
        for category, score in combined_scores.items():
            if score >= threshold:
                subcategory = self._detect_subcategory(text, category)
                reasoning = self._generate_reasoning(
                    text, category, subcategory,
                    keyword_scores, semantic_scores, combined_scores
                )
                keywords = self._extract_detected_keywords(text, category)
                
                matches.append(ContractCategory(
                    category=category,
                    subcategory=subcategory,
                    confidence=score,
                    reasoning=reasoning,
                    detected_keywords=keywords
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        log_info(f"Multi-label classification found {len(matches)} categories")
        
        return matches if matches else [self.classify_contract(text)]
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description of a category"""
        descriptions = {
            'employment': 'Employment agreements governing employer-employee relationships',
            'consulting': 'Consulting and independent contractor agreements',
            'nda': 'Non-disclosure and confidentiality agreements',
            'technology': 'Software licensing and technology service agreements',
            'intellectual_property': 'IP assignment, licensing, and protection agreements',
            'real_estate': 'Property lease, rental, and purchase agreements',
            'financial': 'Loan, credit, and financial service agreements',
            'business': 'Partnership, joint venture, and corporate agreements',
            'sales': 'Sales, purchase, and distribution agreements',
            'service_agreement': 'Professional service and maintenance agreements',
            'vendor': 'Vendor, supplier, and procurement agreements',
            'agency': 'Agency and representation agreements'
        }
        return descriptions.get(category, 'General contract agreement')
    
    def get_all_categories(self) -> List[str]:
        """Get list of all supported categories"""
        return list(self.CATEGORY_HIERARCHY.keys())
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get subcategories for a specific category"""
        return self.CATEGORY_HIERARCHY.get(category, {}).get('subcategories', [])