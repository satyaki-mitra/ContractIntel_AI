# DEPENDENCIES
import re
import sys
import torch
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from sentence_transformers import util

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.risk_rules import ContractType
from config.model_config import ModelConfig
from utils.text_processor import TextProcessor
from utils.logger import ContractAnalyzerLogger
from services.data_models import ContractCategory


class ContractClassifier:
    """
    Contract categorization using:
    1. Legal-BERT embeddings + semantic similarity
    2. Multi-label classification (a contract can be multiple types)
    3. Hierarchical categories (Employment -> Full-Time/Contract/Internship)
    4. Confidence scoring with explanations
    """
    # CATEGORY HIERARCHY WITH KEYWORDS - UPDATED TO MATCH YOUR CATEGORIES
    CATEGORY_HIERARCHY   = {'employment'            : {'subcategories' : ['full_time', 'part_time', 'contract_worker', 'internship', 'executive'],
                                                       'keywords'      : ['employee', 'employment', 'employer', 'job', 'position', 'staff', 'salary', 'wages', 'compensation', 'payroll', 'benefits', 'health insurance', 'retirement', 'pension', '401(k)', 'vacation', 'paid time off', 'sick leave', 'holidays', 'probation', 'performance review', 'promotion', 'termination', 'job description', 'duties', 'responsibilities', 'work hours', 'overtime', 'timekeeping', 'attendance', 'confidentiality', 'non-compete', 'non-solicitation', 'intellectual property', 'inventions', 'work product', 'severance', 'notice period', 'resignation', 'dismissal'],
                                                       'weight'        : 1.1,
                                                      },
                            'consulting'            : {'subcategories' : ['independent_contractor', 'advisory', 'professional_services', 'freelance'],
                                                       'keywords'      : ['consultant', 'consulting', 'independent contractor', 'statement of work', 'deliverables', 'professional services', 'hourly rate', 'project scope', 'milestone', 'acceptance criteria', 'work product', '1099', 'self-employed', 'contractor', 'consulting services', 'expert advice', 'advisory services', 'project basis', 'task order'],
                                                       'weight'        : 1.0,
                                                      },
                            'nda'                   : {'subcategories' : ['mutual_nda', 'unilateral_nda', 'confidentiality_agreement'],
                                                       'keywords'      : ['non-disclosure', 'confidentiality', 'proprietary information', 'nda', 'disclosure agreement', 'trade secret', 'confidential information', 'receiving party', 'disclosing party', 'confidentiality obligation', 'non-use', 'non-circumvention', 'secrecy', 'protected information', 'confidentiality period', 'return of information'],
                                                       'weight'        : 1.0, 
                                                      },
                            'software'              : {'subcategories' : ['software_license', 'saas', 'cloud_services', 'development', 'api_access'],
                                                       'keywords'      : ['software', 'license', 'saas', 'subscription', 'source code', 'object code', 'api', 'cloud', 'hosting', 'maintenance', 'updates', 'support', 'uptime', 'service level', 'software as a service', 'platform', 'application', 'user license', 'perpetual license', 'subscription fee', 'end user license agreement', 'eula'],
                                                       'weight'        : 1.1,
                                                      },
                            'service'               : {'subcategories' : ['master_services', 'maintenance', 'support', 'subscription'],
                                                       'keywords'      : ['service provider', 'services', 'sla', 'service level agreement', 'uptime', 'response time', 'support', 'maintenance', 'service credits', 'performance metrics', 'implementation', 'professional services', 'service description', 'service fees', 'service term', 'service delivery', 'service scope'],
                                                       'weight'        : 1.0,
                                                      },
                            'partnership'           : {'subcategories' : ['business_partnership', 'joint_venture', 'strategic_alliance'],
                                                       'keywords'      : ['partnership', 'joint venture', 'equity', 'shares', 'profit sharing', 'loss allocation', 'management', 'governance', 'voting rights', 'dissolution', 'capital contribution', 'distribution', 'membership interest', 'operating agreement', 'board of directors', 'partnership agreement'],
                                                       'weight'        : 1.0,
                                                      },
                            'lease'                 : {'subcategories' : ['residential_lease', 'commercial_lease', 'sublease', 'equipment_lease'],
                                                       'keywords'      : ['landlord', 'tenant', 'lease', 'premises', 'rent', 'property', 'security deposit', 'utilities', 'maintenance', 'repairs', 'eviction', 'lease term', 'renewal', 'square footage', 'rental agreement', 'lessor', 'lessee', 'property management', 'common areas', 'quiet enjoyment'],
                                                       'weight'        : 1.0,
                                                      },
                            'purchase'              : {'subcategories' : ['asset_purchase', 'stock_purchase', 'goods_purchase'],
                                                       'keywords'      : ['purchase', 'sale', 'buyer', 'seller', 'goods', 'products', 'delivery', 'shipment', 'payment terms', 'invoice', 'purchase price', 'quantity', 'specifications', 'purchase order', 'sales agreement', 'bill of sale', 'title transfer', 'risk of loss', 'closing date'],
                                                       'weight'        : 1.0,
                                                      },
                            'general'               : {'subcategories' : ['standard_agreement', 'basic_contract'],
                                                       'keywords'      : ['agreement', 'contract', 'party', 'parties', 'terms and conditions', 'governing law', 'jurisdiction', 'dispute resolution', 'force majeure', 'notice', 'amendment', 'assignment', 'severability', 'entire agreement'],
                                                       'weight'        : 0.8,
                                                      },
                           } 
    
    # SUBCATEGORY DETECTION PATTERNS
    SUBCATEGORY_PATTERNS = {'full_time'                 : ['full-time', 'full time', 'permanent', 'regular employee', '40 hours', 'exempt employee', 'salary basis'],
                            'part_time'                 : ['part-time', 'part time', 'hours per week', 'non-exempt', 'hourly employee', 'temporary', 'seasonal'],
                            'contract_worker'           : ['independent contractor', 'contract', 'fixed term', 'temporary', 'contract period', 'contract worker', 'contract employee'],
                            'internship'                : ['intern', 'internship', 'student', 'training program', 'educational', 'college credit', 'unpaid intern'],
                            'executive'                 : ['executive', 'ceo', 'cfo', 'cto', 'president', 'vice president', 'director', 'officer', 'executive compensation', 'stock options', 'golden parachute'],
                            'independent_contractor'    : ['independent contractor', '1099', 'contractor', 'self-employed', 'freelance', 'consultant agreement'],
                            'advisory'                  : ['advisor', 'advisory', 'counsel', 'consulting services', 'expert advice', 'advisory board', 'strategic advisory'],
                            'professional_services'     : ['professional services', 'consulting services', 'engagement', 'service provider', 'professional firm'],
                            'freelance'                 : ['freelance', 'freelancer', 'gig', 'project-based', 'freelance work', 'gig economy'],
                            'mutual_nda'                : ['mutual', 'both parties', 'each party', 'reciprocal', 'mutual confidentiality', 'two-way'],
                            'unilateral_nda'            : ['one-way', 'receiving party', 'disclosing party', 'unilateral', 'single party', 'one party'],
                            'confidentiality_agreement' : ['confidentiality agreement', 'secrecy agreement', 'proprietary information agreement'],
                            'software_license'          : ['software license', 'license key', 'perpetual license', 'end user license', 'software agreement'],
                            'saas'                      : ['software as a service', 'saas', 'subscription', 'cloud-based', 'web-based', 'online service'],
                            'cloud_services'            : ['cloud services', 'cloud computing', 'infrastructure', 'iaas', 'paas', 'cloud hosting'],
                            'development'               : ['software development', 'custom development', 'development services', 'programming', 'coding'],
                            'api_access'                : ['api', 'application programming interface', 'api access', 'api key', 'rest api', 'graphql'],
                            'master_services'           : ['master services agreement', 'msa', 'master agreement', 'framework agreement'],
                            'maintenance'               : ['maintenance agreement', 'maintenance services', 'preventive maintenance', 'repair services'],
                            'support'                   : ['support agreement', 'technical support', 'customer support', 'help desk'],
                            'subscription'              : ['subscription agreement', 'subscription service', 'recurring billing', 'subscription fee'],
                            'business_partnership'      : ['partnership', 'general partnership', 'limited partnership', 'partnership agreement'],
                            'joint_venture'             : ['joint venture', 'jv agreement', 'joint venture agreement', 'strategic alliance'],
                            'strategic_alliance'        : ['strategic alliance', 'collaboration agreement', 'cooperation agreement'],
                            'residential_lease'         : ['residential', 'apartment', 'house', 'dwelling', 'residential property', 'tenant', 'landlord', 'rental'],
                            'commercial_lease'          : ['commercial', 'office space', 'retail space', 'commercial property', 'business premises', 'commercial tenant'],
                            'sublease'                  : ['sublease', 'sublet', 'subtenant', 'sublessee', 'sublessor'],
                            'equipment_lease'           : ['equipment lease', 'equipment rental', 'lease equipment', 'leased property'],
                            'asset_purchase'            : ['asset purchase', 'business assets', 'asset sale', 'purchase assets'],
                            'stock_purchase'            : ['stock purchase', 'share purchase', 'equity purchase', 'stock sale'],
                            'goods_purchase'            : ['goods purchase', 'product purchase', 'merchandise', 'inventory purchase'],
                            'standard_agreement'        : ['standard agreement', 'template agreement', 'boilerplate contract'],
                            'basic_contract'            : ['basic contract', 'simple agreement', 'standard terms'],
                           }

    DEFAULT_CONFIDENCE_THRESHOLD = 0.65 
    MULTI_LABEL_THRESHOLD        = 0.55  
                        

    def __init__(self, model_loader):
        """
        Initialize contract classifier
        
        Arguments:
        ----------
            model_loader : ModelLoader instance for accessing Legal-BERT and embeddings
         """
        self.model_loader         = model_loader
        self.embedding_model      = None
        self.legal_bert_model     = None
        self.legal_bert_tokenizer = None
        self.device               = None
        
        # Category template embeddings (computed once)
        self.category_embeddings  = dict()
        
        # Text processor for preprocessing : Don't need spaCy for classification
        self.text_processor       = TextProcessor(use_spacy = False)  
        
        # Logger
        self.logger               = ContractAnalyzerLogger.get_logger()
        
        # Lazy load models
        self._lazy_load()

    
    def _lazy_load(self):
        """
        Lazy load models on first use
        """
        if self.embedding_model is None:
            try:
                log_info("Loading models for contract classification...")
                
                # Load embedding model
                self.embedding_model                             = self.model_loader.load_embedding_model()
                
                # Load Legal-BERT
                self.legal_bert_model, self.legal_bert_tokenizer = self.model_loader.load_legal_bert()
                self.device                                      = self.model_loader.device
                
                # Prepare category embeddings
                self._prepare_category_embeddings()
                
                log_info("Contract classifier models loaded successfully")
                
            except Exception as e:
                log_error(e, context = {"component" : "ContractClassifier", "operation" : "model_loading"})
                raise

    
    def _extract_classification_context(self, full_text: str) -> str:
        """
        Extract key legal sections for more accurate classification
        Focuses on preamble, definitions, and core agreement sections
        
        Arguments:
        ----------
            full_text { str } : Full contract text
        
        Returns:
        --------
                { str }    : Context-rich excerpt for classification
        """
        sections = list()
        
        # First 2000 chars (usually contains parties, effective date, preamble)
        sections.append(full_text[:2000])
        
        # WHEREAS clauses (recitals - explains purpose and background)
        whereas_section = self._extract_section_between(full_text, "WHEREAS", "NOW THEREFORE")
        
        if whereas_section:
            sections.append(whereas_section)
        
        # AGREEMENT section (core contractual terms)
        agreement_section = self._extract_section_between(full_text, "AGREEMENT", "TERMS AND CONDITIONS")
        
        if not agreement_section:
            agreement_section = self._extract_section_containing(full_text, ["AGREES AS FOLLOWS", "HEREBY AGREES"])
        
        if agreement_section:
            sections.append(agreement_section)
        
        # Key definition sections
        definitions_section = self._extract_section_containing(full_text, ["DEFINITIONS", "MEANING OF TERMS"])
        
        if definitions_section:
            sections.append(definitions_section)
        
        # Combine and clean
        context = " ".join([section.strip() for section in sections if section and section.strip()])
        
        # Fallback to original text if context extraction failed
        return context if (len(context) > 500) else full_text


    def _extract_section_between(self, text: str, start_marker: str, end_marker: str) -> Optional[str]:
        """
        Extract text between two markers (case-insensitive)
        """
        try:
            pattern = re.compile(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", re.IGNORECASE | re.DOTALL)
            match   = pattern.search(text)
            
            return match.group(1).strip() if match else None
        
        except Exception:
            return None


    def _extract_section_containing(self, text: str, markers: List[str]) -> Optional[str]:
        """
        Extract section containing any of the markers
        """
        for marker in markers:
            if marker.lower() in text.lower():
                # Extract 500 chars around the marker
                idx   = text.lower().find(marker.lower())
                start = max(0, idx - 250)
                end   = min(len(text), idx + len(marker) + 250)
                
                return text[start:end]
        
        return None

    
    def _prepare_category_embeddings(self):
        """
        Pre-compute embeddings for each category template
        """
        log_info("Preparing category embeddings...")
        
        # More specific templates for each category
        category_templates = {
            'employment': "Employment agreement between employer and employee covering salary benefits job duties work hours vacation sick leave performance reviews termination conditions confidentiality and intellectual property rights",
            'consulting': "Consulting services agreement with independent contractor statement of work deliverables hourly rate project scope milestones acceptance criteria work product ownership and payment terms for professional services",
            'nda': "Non-disclosure agreement protecting confidential information trade secrets proprietary data between parties with confidentiality obligations non-use provisions and return of information requirements",
            'software': "Software license agreement or SaaS subscription for technology services including source code access updates maintenance support service level agreements uptime guarantees and API access",
            'service': "Service level agreement for professional services maintenance support with performance metrics service credits response times uptime guarantees and implementation requirements",
            'partnership': "Business partnership joint venture agreement covering equity shares profit distribution management governance voting rights dissolution terms and capital contributions",
            'lease': "Real estate lease agreement for property rental covering premises description rent payments security deposits maintenance responsibilities utilities and eviction terms",
            'purchase': "Sales purchase agreement for goods products with buyer seller terms covering delivery shipment payment terms invoices purchase price quantity specifications and title transfer",
            'general': "General contract agreement with standard terms and conditions governing law jurisdiction dispute resolution force majeure notice provisions and general legal framework"
        }
        
        for category, template in category_templates.items():
            # Encode template
            embedding                          = self.embedding_model.encode(template, convert_to_tensor = True)
            self.category_embeddings[category] = embedding
        
        log_info(f"Prepared embeddings for {len(self.category_embeddings)} categories")
    

    # MAIN CLASSIFICATION METHOD
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
        if (not contract_text or (len(contract_text) < 100)):
            raise ValueError("Contract text too short for classification")
        
        # Use default threshold if not specified
        if min_confidence is None:
            min_confidence = self.DEFAULT_CONFIDENCE_THRESHOLD

        # Preprocess text 
        text_excerpt = self._extract_classification_context(full_text = contract_text)
        
        log_info("Starting contract classification", 
                 text_length    = len(contract_text),
                 excerpt_length = len(text_excerpt),
                )
        
        # Keyword scoring
        keyword_scores    = self._score_keywords(text_lower = contract_text.lower())

        # Semantic similarity
        semantic_scores   = self._semantic_similarity(text = text_excerpt)

        # Legal-BERT semantic similarity (enhanced)
        legal_bert_scores = self._legal_bert_similarity(text = text_excerpt)

        # Combine scores (weighted average)
        combined_scores   = self._combine_scores(keyword_scores    = keyword_scores,
                                                 semantic_scores   = semantic_scores,
                                                 legal_bert_scores = legal_bert_scores,
                                                )
        
        # Get primary category
        if not combined_scores:
            log_info("No categories detected, defaulting to 'general'")
            return ContractCategory(category          = "general",
                                    subcategory       = None,
                                    confidence        = 0.5,
                                    reasoning         = ["Unable to determine specific contract type"],
                                    detected_keywords = [],
                                   )
        
        primary_category       = max(combined_scores, key = combined_scores.get)
        confidence             = combined_scores[primary_category]
        
        # Detect subcategory
        subcategory            = self._detect_subcategory(text             = contract_text, 
                                                          primary_category = primary_category,
                                                         )
        
        # Generate reasoning
        reasoning              = self._generate_reasoning(contract_text     = contract_text,
                                                          primary_category  = primary_category,
                                                          subcategory       = subcategory,
                                                          keyword_scores    = keyword_scores,
                                                          semantic_scores   = semantic_scores,
                                                          legal_bert_scores = legal_bert_scores,
                                                          combined_scores   = combined_scores,
                                                         )
        
        # Extract detected keywords
        detected_keywords      = self._extract_detected_keywords(contract_text, primary_category)
        
        # Get alternative categories: Top 3 alternatives
        alternative_categories = sorted([(cat, score) for cat, score in combined_scores.items() if cat != primary_category],
                                        key     = lambda x: x[1],
                                        reverse = True,
                                       )[:3]
        
        result                 = ContractCategory(category               = primary_category,
                                                  subcategory            = subcategory,
                                                  confidence             = confidence,
                                                  reasoning              = reasoning,
                                                  detected_keywords      = detected_keywords,
                                                  alternative_categories = alternative_categories,
                                                 )
        
        log_info("Contract classified successfully",
                 category    = primary_category,
                 subcategory = subcategory,
                 confidence  = confidence,
                )
        
        return result
    
    
    def _score_keywords(self, text_lower: str) -> Dict[str, float]:
        """
        Score each category based on keyword presence

        Arguments:
        ----------
            text_lower { str } : Lowercase contract text

        Returns:
        --------
               { dict }        : Dictionary of {category: score}
        """
        scores = dict()
        for category, config in self.CATEGORY_HIERARCHY.items():
            keywords      = config['keywords']
            weight        = config['weight']
            
            # Count keyword matches with partial matching for multi-word terms
            keyword_count = 0
            
            for keyword in keywords:
                # Check for exact match or partial match for multi-word terms
                if ' ' in keyword:
                    # For multi-word terms, check if all words appear in text
                    words = keyword.split()
                    if all(word in text_lower for word in words):
                        keyword_count += 1
                
                else:
                    # For single words, exact word boundary match
                    if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                        keyword_count += 1

            # Normalize by number of keywords and apply weight
            normalized_score = (keyword_count / len(keywords)) * weight
            
            # Cap at 1.0
            scores[category] = min(normalized_score, 1.0)
        
        return scores
    

    def _semantic_similarity(self, text: str) -> Dict[str, float]:
        """
        Calculate semantic similarity to category templates using embeddings
        
        Arguments:
        ----------
            text { str } : Contract text excerpt
        
        Returns:
        --------
            { dict }     : Dictionary of {category: similarity_score}
        """
        # Encode contract text
        text_embedding = self.embedding_model.encode(text, convert_to_tensor = True)
        
        # Calculate similarity to each category
        similarities   = dict()

        for category, cat_embedding in self.category_embeddings.items():
            similarity             = util.cos_sim(text_embedding, cat_embedding)[0][0].item()
            similarities[category] = similarity
        
        return similarities

    
    def _legal_bert_similarity(self, text: str) -> Dict[str, float]:
        """
        Use Legal-BERT for semantic similarity calculation
        
        Arguments:
        ----------
            text { str } : Contract text excerpt
        
        Returns:
        --------
            { dict }     : Dictionary of {category: similarity_score} using Legal-BERT embeddings
        """
        # Get Legal-BERT embedding for the text
        text_embedding = self._get_legal_bert_embedding(text)
        
        # Calculate similarity to each category's Legal-BERT embedding
        similarities   = dict()
        
        for category in self.CATEGORY_HIERARCHY.keys():
            # Get pre-computed category embedding
            cat_embedding          = self._get_legal_bert_embedding(f"This is a {category.replace('_', ' ')} contract agreement")
            
            # Calculate cosine similarity
            similarity             = torch.nn.functional.cosine_similarity(torch.tensor(text_embedding).unsqueeze(0), torch.tensor(cat_embedding).unsqueeze(0)).item()
            
            similarities[category] = similarity
        
        return similarities
    
    
    def _get_legal_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get Legal-BERT embedding for text using [CLS] token
        
        Arguments:
        ----------
            text { str }   : Input text
        
        Returns:
        --------
            { np.ndarray } : Embedding vector
        """
        # Tokenize
        inputs = self.legal_bert_tokenizer(text,
                                           return_tensors = "pt",
                                           padding        = True,
                                           truncation     = True,
                                           max_length     = 512,
                                          ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs       = self.legal_bert_model(**inputs)
            # Use [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return cls_embedding
    

    def _combine_scores(self, keyword_scores: Dict[str, float], semantic_scores: Dict[str, float], legal_bert_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Combine scores from different methods (weighted average)
        
        Arguments:
        ----------
            keyword_scores    { dict } : Keyword-based scores

            semantic_scores   { dict } : Semantic similarity scores
            
            legal_bert_scores { dict } : Legal-BERT similarity scores (optional)
        
        Returns:
        --------
                   { dict }            : Combined scores dictionary
        """
        combined          = dict()
        
        # Weights for each method
        keyword_weight    = 0.35  
        semantic_weight   = 0.35 
        legal_bert_weight = 0.30 
        
        for category in self.CATEGORY_HIERARCHY.keys():
            score = (keyword_scores.get(category, 0) * keyword_weight + 
                     semantic_scores.get(category, 0) * semantic_weight + 
                     legal_bert_scores.get(category, 0) * legal_bert_weight
                    )
            
            combined[category] = score
        
        return combined
    
    
    def _detect_subcategory(self, text: str, primary_category: str) -> Optional[str]:
        """
        Detect specific subcategory within primary category
        
        Arguments:
        ----------
            text             { str } : Full contract text

            primary_category { str } : Detected primary category
        
        Returns:
        --------
                  { str }            : Subcategory name or None
        """
        text_lower    = text.lower()
        
        # Get subcategories for this category
        subcategories = self.CATEGORY_HIERARCHY[primary_category]['subcategories']
        
        # Score each subcategory
        subcat_scores = dict()

        for subcat in subcategories:
            if subcat in self.SUBCATEGORY_PATTERNS:
                patterns              = self.SUBCATEGORY_PATTERNS[subcat]
                score                 = sum(1 for pattern in patterns if pattern in text_lower)
                subcat_scores[subcat] = score
        
        # Return best match if any
        if (subcat_scores and (max(subcat_scores.values()) > 0)):
            best_subcat = max(subcat_scores, key = subcat_scores.get)
            log_info(f"Detected subcategory: {best_subcat}", 
                     category = primary_category,
                     score    = subcat_scores[best_subcat],
                    )

            return best_subcat
        
        return None
    

    def _generate_reasoning(self, contract_text: str, primary_category: str, subcategory: Optional[str], keyword_scores: Dict[str, float], 
                            semantic_scores: Dict[str, float], legal_bert_scores: Dict[str, float], combined_scores: Dict[str, float]) -> List[str]:
        """
        Generate human-readable reasoning for classification
        
        Returns:
        --------
            { list } : List of reasoning statements
        """
        reasoning        = list()
        
        # Primary category reasoning
        keyword_match    = keyword_scores.get(primary_category, 0)
        semantic_match   = semantic_scores.get(primary_category, 0)
        legal_bert_match = legal_bert_scores.get(primary_category, 0)
        
        # Keyword-based reasoning
        if (keyword_match > 0.6):
            reasoning.append(f"Strong keyword indicators for {primary_category.replace('_', ' ')} category ({int(keyword_match * 100)}% keyword match)")

        elif (keyword_match > 0.3):
            reasoning.append(f"Moderate keyword presence for {primary_category.replace('_', ' ')} ({int(keyword_match * 100)}% keyword match)")

        elif (keyword_match > 0.1):
            reasoning.append(f"Limited keyword indicators for {primary_category.replace('_', ' ')} ({int(keyword_match * 100)}% keyword match)")
        
        # Semantic similarity reasoning
        if (semantic_match > 0.70):
            reasoning.append(f"High semantic similarity to {primary_category.replace('_', ' ')} agreements (similarity: {semantic_match:.2f})")

        elif (semantic_match > 0.55):
            reasoning.append(f"Moderate semantic similarity to {primary_category.replace('_', ' ')} contracts (similarity: {semantic_match:.2f})"
                            )
        
        # Legal-BERT reasoning
        if (legal_bert_match > 0.65):
            reasoning.append(f"Legal-BERT analysis strongly supports {primary_category.replace('_', ' ')} classification (similarity: {legal_bert_match:.2f})"
                            )

        elif (legal_bert_match > 0.50):
            reasoning.append(f"Legal-BERT analysis moderately supports {primary_category.replace('_', ' ')} classification (similarity: {legal_bert_match:.2f})"
                            )
        
        # Subcategory reasoning
        if subcategory:
            reasoning.append(f"Specific subcategory identified: {subcategory.replace('_', ' ')}")
        
        # Alternative categories (if close)
        sorted_scores = sorted(combined_scores.items(), key = lambda x: x[1], reverse = True)
        
        if ((len(sorted_scores) > 1) and (sorted_scores[1][1] > 0.30)):
            alt_category, alt_score = sorted_scores[1]
            
            reasoning.append(f"Also contains elements of {alt_category.replace('_', ' ')} (secondary match: {alt_score:.2f})")
        
        # If no strong reasoning
        if not reasoning:
            reasoning.append("Classification based on general contract structure and terminology")
        
        return reasoning

    
    def _extract_detected_keywords(self, text: str, category: str) -> List[str]:
        """
        Extract which specific keywords were found
        
        Arguments:
        ----------
            text     { str } : Contract text

            category { str } : Detected category
        
        Returns:
        --------
                { list }     : List of detected keywords
        """
        text_lower = text.lower()
        keywords   = self.CATEGORY_HIERARCHY[category]['keywords']
        
        detected   = [kw for kw in keywords if kw in text_lower]

        # Return all detected keywords
        return detected
    
    
    @ContractAnalyzerLogger.log_execution_time("classify_multi_label")
    def classify_multi_label(self, text: str, threshold: float = None) -> List[ContractCategory]:
        """
        Classify as multiple categories if applicable (e.g., Employment + NDA, Consulting + IP Assignment)
        
        Arguments:
        ----------
            text       { str }  : Contract text

            threshold { float } : Minimum confidence threshold for multi-label
        
        Returns:
        --------
                 { list }       : List of ContractCategory objects (sorted by confidence)
        """
        # Use multi-label threshold if not specified
        if threshold is None:
            threshold = self.MULTI_LABEL_THRESHOLD

        log_info("Starting multi-label classification", threshold = threshold)
        
        # Get scores
        keyword_scores    = self._score_keywords(text_lower = text.lower())
        semantic_scores   = self._semantic_similarity(text = text)
        legal_bert_scores = self._legal_bert_similarity(text = text)
        combined_scores   = self._combine_scores(keyword_scores    = keyword_scores, 
                                                 semantic_scores   = semantic_scores, 
                                                 legal_bert_scores = legal_bert_scores,
                                                )
        
        # Get all categories above threshold
        matches         = list()

        for category, score in combined_scores.items():
            if (score >= threshold):
                subcategory = self._detect_subcategory(text             = text, 
                                                       primary_category = category,
                                                      )
                                                  
                reasoning   = self._generate_reasoning(contract_text     = text, 
                                                       primary_category  = category, 
                                                       subcategory       = subcategory, 
                                                       keyword_scores    = keyword_scores, 
                                                       semantic_scores   = semantic_scores, 
                                                       legal_bert_scores = legal_bert_scores, 
                                                       combined_scores   = combined_scores,
                                                      )

                keywords    = self._extract_detected_keywords(text     = text, 
                                                              category = category,
                                                             )
                
                matches.append(ContractCategory(category          = category,
                                                subcategory       = subcategory,
                                                confidence        = score,
                                                reasoning         = reasoning,
                                                detected_keywords = keywords,
                                               )
                              )
        
        # Sort by confidence
        matches.sort(key = lambda x: x.confidence, reverse = True)
        
        log_info(f"Multi-label classification found {len(matches)} categories")
        
        return matches if matches else [self.classify_contract(text)]
    

    def get_category_description(self, category: str) -> str:
        """
        Get human-readable description of a category
        """
        descriptions = {'employment'  : 'Employment agreements governing employer-employee relationships',
                        'consulting'  : 'Consulting and independent contractor agreements',
                        'nda'         : 'Non-disclosure and confidentiality agreements',
                        'software'    : 'Software licensing and technology service agreements',
                        'service'     : 'Professional service and maintenance agreements',
                        'partnership' : 'Partnership, joint venture, and corporate agreements',
                        'lease'       : 'Property lease, rental, and equipment lease agreements',
                        'purchase'    : 'Sales, purchase, and goods transfer agreements',
                        'general'     : 'General contract agreements with standard terms and conditions',
                       }

        return descriptions.get(category, 'General contract agreement')

    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all supported categories
        """
        return list(self.CATEGORY_HIERARCHY.keys())
    

    def get_subcategories(self, category: str) -> List[str]:
        """
        Get subcategories for a specific category
        """
        return self.CATEGORY_HIERARCHY.get(category, {}).get('subcategories', [])