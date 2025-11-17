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
from collections import defaultdict
from sentence_transformers import util

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.risk_rules import RiskRules
from config.risk_rules import ContractType
from utils.text_processor import TextProcessor
from utils.logger import ContractAnalyzerLogger
from services.data_models import ExtractedClause
from model_manager.model_loader import ModelLoader
from services.data_models import ClauseInterpretation


class ComprehensiveClauseExtractor:
    """
    For general clause extraction across all contract types : Extracts and classifies clauses using Legal-BERT + structural patterns
    
    Will be used for: General document analysis, clause discovery, contract understanding
    """
    # COMPREHENSIVE CLAUSE CATEGORIES COVERING ALL LEGAL AREAS
    CLAUSE_CATEGORIES = {'compensation'          : {'keywords'            : ['salary', 'wage', 'compensation', 'pay', 'payment', 'bonus', 'commission', 'remuneration', 'fee', 'rate', 'benefits', 'equity', 'stock options', 'incentive'],
                                                    'representative_text' : ("The Employee shall receive an annual base salary of One Hundred Thousand Dollars payable in accordance with the Company's standard payroll practices. Additional compensation may include performance bonuses and stock options."),
                                                    'weight'              : 1.0,
                                                   },
                         'termination'           : {'keywords'            : ['termination', 'terminate', 'notice period', 'resignation', 'dismissal', 'severance', 'end of employment', 'cessation', 'notice', 'for cause', 'without cause'],
                                                    'representative_text' : ("Either party may terminate this Agreement upon thirty days written notice. The Company may terminate for cause immediately upon written notice to Employee. Upon termination, Employee shall receive severance compensation."),
                                                    'weight'              : 1.2,
                                                   },
                         'non_compete'           : {'keywords'            : ['non-compete', 'non-solicit', 'non-solicitation', 'restrictive covenant', 'competitive', 'competition', 'competing business', 'competitive activities', 'non-competition'],
                                                    'representative_text' : ("Employee agrees not to engage in any competitive business activities for a period of twelve months following termination within a fifty-mile radius. Employee shall not solicit Company clients or employees during this period."),
                                                    'weight'              : 1.5,
                                                   },
                         'confidentiality'       : {'keywords'            : ['confidential', 'proprietary', 'trade secret', 'disclosure', 'confidentiality', 'secret', 'private', 'non-disclosure', 'protected information'],
                                                    'representative_text' : ("Employee shall maintain the confidentiality of all proprietary information and trade secrets of the Company. Confidential Information includes business plans, customer lists, and technical data. These obligations survive termination."),
                                                    'weight'              : 1.1,
                                                   },
                         'indemnification'       : {'keywords'            : ['indemnify', 'indemnification', 'hold harmless', 'defend', 'liability', 'claims', 'losses', 'damages', 'indemnity'],
                                                    'representative_text' : ("Party A shall indemnify and hold harmless Party B from any claims, losses, or damages arising from Party A's breach or negligence. This indemnification includes reasonable attorneys' fees and costs of defense."),
                                                    'weight'              : 1.3,
                                                   },
                         'intellectual_property' : {'keywords'            : ['intellectual property', 'ip', 'copyright', 'patent', 'trademark', 'work product', 'inventions', 'creation', 'ownership', 'ip rights', 'proprietary rights'],
                                                    'representative_text' : ("All work product and inventions created by Employee during employment shall be the exclusive property of the Company. Employee assigns all intellectual property rights including patents, copyrights, and trade secrets to the Company."),
                                                    'weight'              : 1.2,
                                                   },
                         'liability'             : {'keywords'            : ['liable', 'liability', 'damages', 'limitation', 'consequential', 'indirect', 'punitive', 'cap', 'limited liability', 'responsibility'],
                                                    'representative_text' : ("In no event shall either party be liable for indirect, incidental, or consequential damages. Total liability under this Agreement shall not exceed the amounts paid in the twelve months preceding the claim."),
                                                    'weight'              : 1.2,
                                                   },
                         'warranty'              : {'keywords'            : ['warranty', 'warrant', 'representation', 'guarantee', 'assurance', 'promise', 'warranties', 'guaranty'],
                                                    'representative_text' : ("Company warrants that the Services will be performed in a professional manner. EXCEPT AS EXPRESSLY PROVIDED, COMPANY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE."),
                                                    'weight'              : 0.9,
                                                   },
                         'dispute_resolution'    : {'keywords'            : ['arbitration', 'mediation', 'dispute', 'jurisdiction', 'governing law', 'venue', 'forum', 'resolution', 'litigation'],
                                                    'representative_text' : ("Any disputes arising under this Agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association. This Agreement shall be governed by the laws of the State of California."),
                                                    'weight'              : 0.9,
                                                   },
                         'insurance'             : {'keywords'            : ['insurance', 'coverage', 'insured', 'policy', 'premium', 'insurer', 'liability insurance'],
                                                    'representative_text' : ("Contractor shall maintain general liability insurance with minimum coverage of one million dollars per occurrence. Proof of insurance shall be provided to Client. Company shall be named as additional insured on all policies."),
                                                    'weight'              : 0.8,
                                                   },
                         'assignment'            : {'keywords'            : ['assignment', 'assign', 'transfer', 'successor', 'binding', 'assignee', 'assignor'],
                                                    'representative_text' : ("This Agreement may not be assigned by either party without the prior written consent of the other party. This Agreement shall be binding upon and inure to the benefit of the parties' successors and permitted assigns."),
                                                    'weight'              : 0.8,
                                                   },
                         'amendment'             : {'keywords'            : ['amendment', 'modify', 'modification', 'change', 'alteration', 'waiver', 'amend'],
                                                    'representative_text' : ("This Agreement may not be amended or modified except by written instrument signed by both parties. No waiver of any provision shall be effective unless in writing. All modifications must be mutually agreed upon."),
                                                    'weight'              : 0.7,
                                                   },
                         'force_majeure'         : {'keywords'            : ['force majeure', 'act of god', 'unforeseeable', 'beyond control', 'natural disaster', 'unforeseen circumstances'],
                                                    'representative_text' : ("Neither party shall be liable for failure to perform due to causes beyond its reasonable control including acts of God, war, strikes, or natural disasters. Performance shall be suspended during the force majeure event."),
                                                    'weight'              : 0.7,
                                                   },
                         'entire_agreement'      : {'keywords'            : ['entire agreement', 'integration', 'supersedes', 'prior agreements', 'complete agreement', 'whole agreement'],
                                                    'representative_text' : ("This Agreement constitutes the entire agreement between the parties and supersedes all prior agreements, whether written or oral. No other representations or warranties shall be binding unless incorporated herein."),
                                                    'weight'              : 0.6,
                                                   },
                         'payment_terms'         : {'keywords'            : ['payment terms', 'net 30', 'due date', 'invoice', 'billing', 'payment due', 'late payment', 'interest'],
                                                    'representative_text' : ("Payment shall be due within thirty days of invoice date. Late payments shall accrue interest at the rate of 1.5% per month. All payments shall be made in US dollars."),
                                                    'weight'              : 0.9,
                                                   },
                         'governing_law'         : {'keywords'            : ['governing law', 'jurisdiction', 'venue', 'applicable law', 'state law', 'federal law'],
                                                    'representative_text' : ("This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware. Any legal action shall be brought in the state or federal courts located in Wilmington, Delaware."),
                                                    'weight'              : 0.8,
                                                   },
                         'general'               : {'keywords'            : ['provision', 'term', 'condition', 'obligation', 'requirement', 'clause', 'section'],
                                                    'representative_text' : ("The parties agree to the following terms and conditions governing their relationship. Each party shall perform its obligations in good faith and in accordance with industry standards and applicable law."),
                                                    'weight'              : 0.5,
                                                   }
                        }
    

    def __init__(self, model_loader: ModelLoader):
        """
        Initialize comprehensive clause extractor
        
        Arguments:
        ----------
            model_loader { ModelLoader } : ModelLoader instance for accessing Legal-BERT
        """
        self.model_loader         = model_loader
        
        # Models (lazy loaded)
        self.legal_bert_model     = None
        self.legal_bert_tokenizer = None
        self.embedding_model      = None
        self.device               = None
        
        # Category embeddings (computed from representative texts)
        self.category_embeddings  = dict()
        
        # Text processor
        self.text_processor       = TextProcessor(use_spacy = False)
        
        # Logger
        self.logger               = ContractAnalyzerLogger.get_logger()
        
        # Lazy load
        self._lazy_load()

        # Risk Rules 
        self.risk_rules           = RiskRules()
    

    def _lazy_load(self):
        """
        Lazy load Legal-BERT and embedding models
        """
        if self.legal_bert_model is None:
            try:
                log_info("Loading Legal-BERT for comprehensive clause extraction...")
                
                # Load Legal-BERT (nlpaueb/legal-bert-base-uncased)
                self.legal_bert_model, self.legal_bert_tokenizer = self.model_loader.load_legal_bert()
                self.device                                      = self.model_loader.device
                
                # Load sentence transformer for embeddings
                self.embedding_model                             = self.model_loader.load_embedding_model()
                
                # Prepare category embeddings using Legal-BERT
                self._prepare_category_embeddings()
                
                log_info("Comprehensive clause extractor models loaded successfully")
                
            except Exception as e:
                log_error(e, context = {"component": "ComprehensiveClauseExtractor", "operation": "model_loading"})
                raise

    
    def _prepare_category_embeddings(self):
        """
        Pre-compute Legal-BERT embeddings for category representative texts
        """
        log_info("Computing Legal-BERT embeddings for comprehensive clause categories...")
        
        for category, config in self.CLAUSE_CATEGORIES.items():
            representative_text                = config['representative_text']
            
            # Get Legal-BERT embedding (using [CLS] token)
            embedding                          = self._get_legal_bert_embedding(text = representative_text)

            self.category_embeddings[category] = embedding
        
        log_info(f"Prepared Legal-BERT embeddings for {len(self.category_embeddings)} categories")
    

    def _get_legal_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get Legal-BERT embedding for text using [CLS] token
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
    

    @ContractAnalyzerLogger.log_execution_time("extract_clauses")
    def extract_clauses(self, contract_text: str, max_clauses: int = 25) -> List[ExtractedClause]:
        """
        Extract and classify clauses from contract using hybrid approach
        
        Process:
        1. Structural extraction (numbered sections)
        2. Semantic chunking (for unstructured text)
        3. Legal-BERT classification
        4. Deduplicate and rank by confidence
        
        Arguments:
        ----------
            contract_text { str } : Full contract text

            max_clauses   { int } : Maximum number of clauses to return
        
        Returns:
        --------
                 { list }         : List of ExtractedClause objects sorted by confidence
        """
        
        log_info("Starting comprehensive clause extraction", 
                 text_length = len(contract_text),
                 max_clauses = max_clauses,
                )
        
        # Extract using structural patterns
        structural_clauses = self._extract_structural_clauses(contract_text)
        log_info(f"Extracted {len(structural_clauses)} structural clauses")
        
        # Semantic chunking for unstructured parts
        semantic_chunks    = self._semantic_chunking(contract_text, structural_clauses)
        log_info(f"Created {len(semantic_chunks)} semantic chunks")
        
        # Combine all candidates
        all_candidates     = structural_clauses + semantic_chunks
        log_info(f"Total candidates: {len(all_candidates)}")
        
        # Classify with Legal-BERT
        classified_clauses = self._classify_clauses_with_legal_bert(all_candidates)
        log_info(f"Classified {len(classified_clauses)} clauses")
        
        # Deduplicate and rank
        final_clauses      = self._deduplicate_and_rank(classified_clauses, max_clauses)
        log_info(f"Final output: {len(final_clauses)} clauses")
        
        return final_clauses
    

    def generate_clause_analysis(self, clause: ExtractedClause, llm_interpretation: ClauseInterpretation = None) -> Dict[str, str]:
        """
        Generate analysis and recommendation for a clause
        
        Arguments:
        ----------
            clause               { ExtractedClause }    : ExtractedClause object

            llm_interpretation { ClauseInterpretation } : Optional ClauseInterpretation from LLM
        
        Returns:
        --------
                           { dict }                     : Dictionary with 'analysis' and 'recommendation' keys
        """
        if llm_interpretation:
            # Use LLM interpretation if available
            analysis = llm_interpretation.plain_english_summary
            
            # Combine key points into analysis
            if llm_interpretation.key_points:
                analysis += " " + " ".join(llm_interpretation.key_points[:2])
            
            # Combine potential risks into analysis
            if llm_interpretation.potential_risks:
                risk_text = " Key risks: " + ", ".join(llm_interpretation.potential_risks[:2])
                analysis += risk_text
            
            # Use suggested improvements as recommendation
            if llm_interpretation.suggested_improvements:
                recommendation = " ".join(llm_interpretation.suggested_improvements[:2])
            
            else:
                recommendation = "Review this clause with legal counsel for specific recommendations."
        
        else:
            # Fallback: Generate analysis from risk indicators and category
            risk_indicators = clause.risk_indicators if clause.risk_indicators else []
            risk_score      = getattr(clause, 'risk_score', 0)
            
            # Generate specific analysis based on category and risk
            analysis        = self._generate_fallback_analysis(clause          = clause, 
                                                               risk_indicators = risk_indicators, 
                                                               risk_score      = risk_score,
                                                              )

            recommendation  = self._generate_fallback_recommendation(clause          = clause, 
                                                                     risk_indicators = risk_indicators, 
                                                                     risk_score      = risk_score,
                                                                    )
        
        return {'analysis'       : analysis,
                'recommendation' : recommendation,
               }


    def _generate_fallback_analysis(self, clause: ExtractedClause, risk_indicators: List[str], risk_score: float) -> str:
        """
        Generate fallback analysis when LLM unavailable
        """
        category_analyses = {'compensation'          : f"This compensation clause {'contains concerning terms' if risk_score > 50 else 'appears standard'} regarding payment obligations and structures. ",
                             'termination'           : f"This termination clause {'creates significant imbalance' if risk_score > 60 else 'establishes'} the conditions and procedures for ending the agreement. ",
                             'non_compete'           : f"This restrictive covenant {'is overly broad and' if risk_score > 60 else ''} limits future business activities and employment opportunities. ",
                             'confidentiality'       : f"This confidentiality provision {'has excessive scope' if risk_score > 50 else 'defines'} the obligations to protect sensitive information. ",
                             'indemnification'       : f"This indemnification clause {'creates one-sided liability exposure' if risk_score > 60 else 'allocates'} responsibility for claims and losses. ",
                             'intellectual_property' : f"This IP clause {'may claim overly broad ownership' if risk_score > 50 else 'addresses'} rights to work product and inventions. ",
                             'liability'             : f"This liability provision {'lacks adequate caps or limitations' if risk_score > 60 else 'establishes'} the financial exposure for damages. ",
                            }
        
        analysis          = category_analyses.get(clause.category, f"This {clause.category} clause establishes specific rights and obligations. ")
        
        # Add risk-specific details
        if risk_indicators:
            analysis += f"Specific concerns include: {', '.join(risk_indicators[:3])}. "
        
        if (risk_score > 70):
            analysis += "This clause requires immediate attention and likely modification."

        elif (risk_score > 50):
            analysis += "This clause should be reviewed carefully and potentially negotiated."

        else:
            analysis += "This clause appears to contain standard provisions for this type of agreement."
        
        return analysis


    def _generate_fallback_recommendation(self, clause: ExtractedClause, risk_indicators: List[str], risk_score: float) -> str:
        """
        Generate fallback recommendation when LLM unavailable
        """
        if (risk_score > 70):
            return f"Strongly recommend negotiating substantial changes to this clause. Seek legal counsel to address the identified risks and ensure your interests are protected."
        
        elif (risk_score > 50):
            return f"Negotiate modifications to balance the terms more fairly. Consider adding protective language or limiting the scope of obligations."
        
        elif (risk_score > 30):
            return f"Review with legal counsel to ensure the terms are clear and acceptable. Minor clarifications may be beneficial."
        
        else:
            return f"Standard clause - review for consistency with the overall agreement and your business needs."

    
    def _extract_structural_clauses(self, text: str) -> List[Dict]:
        """
        Extract clauses using structural numbering patterns
        """
        candidates = list()
        
        # Clean text
        text       = re.sub(r'\s+', ' ', text)
        
        # Patterns for legal numbering
        patterns   = [(r'(\d+\.\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=\d+\.\d+(?:\.\d+)*\.|$)', 'numbered'),
                      (r'(Article\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+))\.\s*([^\n]{30,800}?)(?=Article\s+(?:\d+|[IVXLCDM]+)|$)', 'article'),
                      (r'(Section\s+\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=Section\s+\d+|$)', 'section'),
                      (r'(Clause\s+\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=Clause\s+\d+|$)', 'clause'),
                      (r'\(([a-z]|[ivxlcdm]+)\)\s*([^\n]{30,500}?)(?=\([a-z]|[ivxlcdm]+\)|\n\n|$)', 'subclause'),
                     ]
        
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)

            for match in matches:
                clause_text = match.group(2).strip()
                
                # Filter out boilerplate/definitions
                if not self._is_boilerplate(clause_text):
                    # Check for meaningful content
                    if self._has_meaningful_content(clause_text):
                        candidates.append({'text'      : clause_text,
                                           'reference' : match.group(1).strip(),
                                           'start'     : match.start(),
                                           'end'       : match.end(),
                                           'type'      : 'structural',
                                           'ref_type'  : ref_type,
                                         })
        
        # Remove overlapping clauses
        candidates = self._remove_overlapping(candidates)
        
        return candidates

    
    def _is_boilerplate(self, text: str) -> bool:
        """
        Check if text is boilerplate/definitional rather than substantive
        """
        boilerplate_indicators = ['shall mean', 
                                  'means and includes', 
                                  'defined as', 
                                  'definition of',
                                  'hereinafter referred to', 
                                  'for purposes of this', 
                                  'interpretation of',
                                  'as used in this', 
                                  'the term', 
                                  'shall include', 
                                  'includes but not limited',
                                 ]
        
        text_lower             = text.lower()
        # Must have at least one strong indicator AND be definition-heavy
        has_indicator          = any(indicator in text_lower for indicator in boilerplate_indicators)
        is_short_definition    = len(text.split()) < 50 and '"' in text
        
        return has_indicator or is_short_definition
    

    def _has_meaningful_content(self, text: str) -> bool:
        """
        Check if text has meaningful legal content
        """
        # Must have minimum length
        if (len(text.split()) < 15):
            return False
        
        # Check for legal action verbs
        action_verbs   = ['shall', 
                          'must', 
                          'will', 
                          'may', 
                          'agrees', 
                          'undertakes', 
                          'covenants', 
                          'warrants', 
                          'represents', 
                          'acknowledges', 
                          'certifies', 
                          'indemnifies', 
                          'waives', 
                          'terminates',
                         ]
        
        text_lower     = text.lower()
        has_action     = any(verb in text_lower for verb in action_verbs)
        
        # Check for legal subjects
        legal_subjects = ['party', 
                          'parties', 
                          'employee', 
                          'employer', 
                          'company', 
                          'contractor', 
                          'consultant', 
                          'client',
                          'vendor', 
                          'buyer', 
                          'seller', 
                          'landlord', 
                          'tenant', 
                          'licensor', 
                          'licensee',
                         ]
        
        has_subject    = any(subj in text_lower for subj in legal_subjects)
        
        return has_action or has_subject
    

    def _remove_overlapping(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove overlapping clause extractions
        """
        if not candidates:
            return []
        
        # Sort by start position
        candidates.sort(key = lambda x: x['start'])
        
        non_overlapping = [candidates[0]]
        
        for candidate in candidates[1:]:
            last = non_overlapping[-1]
            
            # Check if overlaps
            if (candidate['start'] >= last['end']):
                non_overlapping.append(candidate)

            elif (len(candidate['text']) > len(last['text'])):
                # Keep longer clause if overlapping
                non_overlapping[-1] = candidate
        
        return non_overlapping
    
    
    def _semantic_chunking(self, text: str, structural_clauses: List[Dict], chunk_size: int = 200) -> List[Dict]:
        """
        Chunk unstructured text semantically uses sentence boundaries
        """
        # Get covered ranges from structural clauses
        covered_ranges = [(c['start'], c['end']) for c in structural_clauses]
        
        # Split into sentences
        sentences      = self.text_processor.extract_sentences(text)
        
        chunks         = list()
        current_chunk  = list()
        current_length = 0
        current_start  = 0
        
        for sentence in sentences:
            # Check if sentence is already covered by structural extraction
            sentence_start = text.find(sentence, current_start)
            if (sentence_start == -1):
                continue
                
            if self._is_in_range(sentence_start, covered_ranges):
                current_start = sentence_start + len(sentence)
                continue
            
            current_chunk.append(sentence)
            current_length += len(sentence.split())
            
            # Create chunk when reaching size limit
            if (current_length >= chunk_size):
                chunk_text = ' '.join(current_chunk).strip()
                
                if (len(chunk_text) >= 50) and (not self._is_boilerplate(chunk_text)):
                    if self._has_meaningful_content(chunk_text):
                        chunks.append({'text'      : chunk_text,
                                       'reference' : f'Semantic-{len(chunks)+1}',
                                       'start'     : sentence_start,
                                       'end'       : sentence_start + len(chunk_text),
                                       'type'      : 'semantic',
                                       'ref_type'  : 'semantic',
                                     })
                
                current_chunk  = list()
                current_length = 0
            
            current_start = sentence_start + len(sentence)
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            
            if ((len(chunk_text) >= 50) and (not self._is_boilerplate(chunk_text))):
                if self._has_meaningful_content(chunk_text):
                    sentence_start = text.find(current_chunk[0])
                    chunks.append({'text'      : chunk_text,
                                   'reference' : f'Semantic-{len(chunks)+1}',
                                   'start'     : sentence_start,
                                   'end'       : sentence_start + len(chunk_text),
                                   'type'      : 'semantic',
                                   'ref_type'  : 'semantic',
                                 })
        
        return chunks
    

    def _is_in_range(self, position: int, ranges: List[Tuple[int, int]]) -> bool:
        """
        Check if position is within any of the ranges
        """
        return any(start <= position <= end for start, end in ranges)
    
    
    def _classify_clauses_with_legal_bert(self, candidates: List[Dict]) -> List[ExtractedClause]:
        """
        Classify clauses using Legal-BERT embeddings + keyword matching
        """
        classified = list()
        
        for candidate in candidates:
            # Get Legal-BERT embedding for clause
            clause_embedding                       = self._get_legal_bert_embedding(text = candidate['text'])
            
            # Classify using hybrid approach
            category, confidence, legal_bert_score = self._classify_single_clause(text             = candidate['text'], 
                                                                                  clause_embedding = clause_embedding,
                                                                                 )
            
            # Extract risk indicators
            risk_indicators                        = self._extract_risk_indicators(text = candidate['text'])
            
            # Extract sub-clauses if any
            subclauses                             = self._extract_subclauses(text = candidate['text'])
            
            classified.append(ExtractedClause(text              = candidate['text'],
                                              reference         = candidate['reference'],
                                              category          = category,
                                              confidence        = confidence,
                                              start_pos         = candidate['start'],
                                              end_pos           = candidate['end'],
                                              extraction_method = candidate['type'],
                                              risk_indicators   = risk_indicators,
                                              embeddings        = clause_embedding,
                                              subclauses        = subclauses,
                                              legal_bert_score  = legal_bert_score,
                                             )
                             )
        
        return classified
    

    def _classify_single_clause(self, text: str, clause_embedding: np.ndarray) -> Tuple[str, float, float]:
        """
        Classify single clause using Legal-BERT + keyword matching
        """
        text_lower     = text.lower()
        
        # Keyword matching
        keyword_scores = dict()

        for category, config in self.CLAUSE_CATEGORIES.items():
            keywords                 = config['keywords']
            weight                   = config['weight']
            
            keyword_count            = sum(1 for kw in keywords if kw in text_lower)
            keyword_scores[category] = (keyword_count / len(keywords)) * weight
        
        # Legal-BERT semantic similarity
        semantic_scores         = dict()
        clause_embedding_tensor = torch.tensor(clause_embedding).unsqueeze(0)
        
        for category, category_embedding in self.category_embeddings.items():
            category_embedding_tensor = torch.tensor(category_embedding).unsqueeze(0)
            similarity                = torch.nn.functional.cosine_similarity(clause_embedding_tensor, category_embedding_tensor).item()
            semantic_scores[category] = similarity
        
        # Combine scores (70% semantic, 30% keyword)
        combined_scores = dict()

        for category in self.CLAUSE_CATEGORIES.keys():
            combined                  = (semantic_scores.get(category, 0) * 0.70 + keyword_scores.get(category, 0) * 0.30)
            combined_scores[category] = combined
        
        # Get best category
        best_category    = max(combined_scores, key = combined_scores.get)
        confidence       = combined_scores[best_category]
        legal_bert_score = semantic_scores[best_category]
        
        return best_category, confidence, legal_bert_score
    

    def _extract_risk_indicators(self, text: str) -> List[str]:
        """
        Extract risk indicator keywords from clause text using RiskRule with the central risk rules
        """
        text_lower      = text.lower()
        risk_indicators = list()

        # Check for matches against CRITICAL_KEYWORDS from RiskRules
        for keyword in self.risk_rules.CRITICAL_KEYWORDS.keys():
            if keyword in text_lower:
                risk_indicators.append(keyword)

        # Check for matches against HIGH_RISK_KEYWORDS from RiskRules
        for keyword in self.risk_rules.HIGH_RISK_KEYWORDS.keys():
            if keyword in text_lower:
                risk_indicators.append(keyword)

        # Check for matches against MEDIUM_RISK_KEYWORDS from RiskRules
        for keyword in self.risk_rules.MEDIUM_RISK_KEYWORDS.keys():
            if keyword in text_lower:
                risk_indicators.append(keyword)

        # Check for matches against RISKY_PATTERNS from RiskRules
        for pattern, score, description in self.risk_rules.RISKY_PATTERNS:
            if re.search(pattern, text_lower):
                # Use the description from RiskRules as the indicator
                risk_indicators.append(description)

        # Remove duplicates while preserving order
        seen              = set()
        unique_indicators = list()

        for indicator in risk_indicators:
            if indicator not in seen:
                seen.add(indicator)
                unique_indicators.append(indicator)
        
        return unique_indicators
    

    def _extract_subclauses(self, text: str) -> List[str]:
        """
        Extract sub-clauses from main clause (e.g., (a), (b), (i), (ii))
        """
        # Pattern for sub-clauses: (a), (i), etc.
        subclause_pattern = r'\(([a-z]|[ivxlcdm]+)\)\s*([^()]{20,200}?)(?=\([a-z]|[ivxlcdm]+\)|$)'
        matches           = re.findall(subclause_pattern, text, re.IGNORECASE)
        
        subclauses        = list()

        for ref, subtext in matches:
            clean_text = subtext.strip()
            
            if (len(clean_text) >= 20):
                subclauses.append(f"({ref}) {clean_text}")
        
        # Max 25 sub-clauses
        return subclauses[:25]  
    
    
    def _deduplicate_and_rank(self, clauses: List[ExtractedClause], max_clauses: int) -> List[ExtractedClause]:
        """
        Remove duplicates and rank by confidence + legal_bert_score
        """
        if not clauses:
            return []
        
        # Sort by combined score (confidence * 0.6 + legal_bert_score * 0.4)
        clauses.sort(key = lambda x: (x.confidence * 0.6 + x.legal_bert_score * 0.4), reverse = True)
        
        # Deduplicate by text similarity
        unique_clauses = list()
        seen_texts     = set()
        
        for clause in clauses:
            # Simple deduplication by first 100 chars
            text_key     = clause.text[:100].lower().strip()
            
            # Also check similarity to already added clauses
            is_duplicate = False
            
            for existing in unique_clauses:
                similarity = self._text_similarity(clause.text, existing.text)
                if (similarity > 0.85):
                    is_duplicate = True
                    break
            
            if text_key not in seen_texts and not is_duplicate:
                unique_clauses.append(clause)
                seen_texts.add(text_key)
                
                if (len(unique_clauses) >= max_clauses):
                    break
        
        return unique_clauses
    

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity (simple Jaccard similarity)
        """
        words1       = set(text1.lower().split())
        words2       = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union        = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    
    def get_category_distribution(self, clauses: List[ExtractedClause]) -> Dict[str, int]:
        """
        Get distribution of clause categories
        """
        distribution = defaultdict(int)
        
        for clause in clauses:
            distribution[clause.category] += 1
        
        log_info("Clause category distribution", distribution=dict(distribution))
        
        return dict(distribution)
    

    def get_high_risk_clauses(self, clauses: List[ExtractedClause]) -> List[ExtractedClause]:
        """
        Get clauses with risk indicators
        """
        risky = [c for c in clauses if c.risk_indicators]

        risky.sort(key = lambda x: len(x.risk_indicators), reverse = True)
        
        top_25_risky_clauses = risky[:25]

        return top_25_risky_clauses





class RiskClauseExtractor:
    """
    Risk-Focused Clause Extractor: Specifically for risk analysis using RiskRules framework for contract-type specific risk assessment
    
    This will be used for: Risk analysis, protection gap detection, contract-type specific assessment
    """
    def __init__(self, model_loader: ModelLoader, contract_type: ContractType):
        """
        Initialize risk-focused clause extractor
        
        Arguments:
        ----------
            model_loader  { ModelLoader }  : ModelLoader instance

            contract_type { ContractType } : Contract type for risk rule adjustments
        """
        self.model_loader             = model_loader
        self.contract_type            = contract_type
        self.risk_rules               = RiskRules()
        
        # Models (lazy loaded)
        self.legal_bert_model         = None
        self.legal_bert_tokenizer     = None
        self.embedding_model          = None
        self.device                   = None
        
        # Risk category embeddings
        self.risk_category_embeddings = dict()
        
        # Text processor
        self.text_processor           = TextProcessor(use_spacy = False)
        
        # Logger
        self.logger                   = ContractAnalyzerLogger.get_logger()
        
        # Contract-type specific weights
        self.category_weights         = self.risk_rules.get_adjusted_weights(contract_type)
         
        # Lazy load
        self._lazy_load()
    

    def _lazy_load(self):
        """
        Lazy load models for risk analysis
        """
        if self.legal_bert_model is None:
            try:
                log_info("Loading models for risk-focused clause extraction...")
                
                # Load Legal-BERT
                self.legal_bert_model, self.legal_bert_tokenizer = self.model_loader.load_legal_bert()
                self.device                                      = self.model_loader.device
                
                # Load embedding model
                self.embedding_model                             = self.model_loader.load_embedding_model()
                
                # Prepare risk category embeddings
                self._prepare_risk_category_embeddings()
                
                log_info("Risk clause extractor models loaded successfully")
                
            except Exception as e:
                log_error(e, context = {"component": "RiskClauseExtractor", "operation": "model_loading"})
                raise

    
    def _prepare_risk_category_embeddings(self):
        """
        Prepare embeddings for risk categories using RiskRules framework
        """
        log_info("Preparing risk category embeddings...")
        
        # Create representative texts for each risk category
        risk_category_texts = {'restrictive_covenants' : "Non-compete non-solicitation restrictive covenants competition limitations duration geographic scope industry restrictions",
                               'termination_rights'    : "Termination notice period severance for cause without cause immediate termination at-will employment end of agreement",
                               'penalties_liability'   : "Penalties liquidated damages liability limitations unlimited liability consequential damages indemnification hold harmless",
                               'compensation_benefits' : "Compensation salary wages benefits bonus commission equity stock options retirement health insurance paid time off",
                               'intellectual_property' : "Intellectual property IP ownership copyright patent trademark trade secrets work product inventions proprietary rights",
                               'confidentiality'       : "Confidentiality non-disclosure proprietary information trade secrets protection secrecy confidential obligations",
                               'liability_indemnity'   : "Liability indemnification hold harmless defense costs claims losses damages responsibility accountability",
                               'governing_law'         : "Governing law jurisdiction venue dispute resolution arbitration mediation legal forum applicable law",
                               'payment_terms'         : "Payment terms due date invoice billing net 30 late payment interest fees compensation remuneration",
                               'warranties'            : "Warranties representations guarantees disclaimers merchantability fitness for purpose product quality service standards",
                               'dispute_resolution'    : "Dispute resolution arbitration mediation litigation legal proceedings costs attorneys fees jurisdiction",
                               'assignment_change'     : "Assignment transfer change control amendment modification consent approval successor parties",
                               'insurance'             : "Insurance coverage liability insurance professional indemnity proof of insurance additional insured policy requirements",
                               'force_majeure'         : "Force majeure act of god unforeseen circumstances beyond control natural disasters performance suspension"
                              }
        
        for category, text in risk_category_texts.items():
            embedding                               = self._get_legal_bert_embedding(text)
            self.risk_category_embeddings[category] = embedding
        
        log_info(f"Prepared risk embeddings for {len(self.risk_category_embeddings)} categories")
    

    def _get_legal_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get Legal-BERT embedding for risk analysis
        """
        inputs = self.legal_bert_tokenizer(text,
                                           return_tensors = "pt",
                                           padding        = True,
                                           truncation     = True,
                                           max_length     = 512,
                                          ).to(self.device)
        
        with torch.no_grad():
            outputs       = self.legal_bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return cls_embedding
    

    @ContractAnalyzerLogger.log_execution_time("extract_risk_clauses")
    def extract_risk_clauses(self, contract_text: str, max_clauses: int = 20) -> List[ExtractedClause]:
        """
        Extract clauses specifically for risk analysis using RiskRules framework
        
        Process:
        1. Focus on high-weight categories for this contract type
        2. Use risk patterns from RiskRules
        3. Calculate risk scores based on RiskRules factors
        4. Prioritize clauses with high risk indicators
        
        Arguments:
        ----------
            contract_text { str } : Full contract text

            max_clauses   { int } : Maximum clauses to return
        
        Returns:
        --------
                 { list }         : Risk-focused clauses with calculated risk scores
        """
        log_info("Starting risk-focused clause extraction",
                 contract_type = self.contract_type.value,
                 max_clauses   = max_clauses,
                )
        
        # Use comprehensive extractor as base
        comprehensive_extractor = ComprehensiveClauseExtractor(self.model_loader)
        all_clauses             = comprehensive_extractor.extract_clauses(contract_text = contract_text, 
                                                                          max_clauses   = 50,
                                                                         )
        
        # Re-classify using risk framework
        risk_clauses            = self._reclassify_with_risk_framework(clauses = all_clauses)
        
        # Calculate risk scores
        risk_clauses            = self._calculate_risk_scores(clauses = risk_clauses)
        
        # Prioritize by risk score and contract-type relevance
        prioritized             = self._prioritize_risk_clauses(clauses     = risk_clauses, 
                                                                max_clauses = max_clauses,
                                                               )
        
        log_info(f"Extracted {len(prioritized)} risk-focused clauses")
        
        return prioritized
    

    def _reclassify_with_risk_framework(self, clauses: List[ExtractedClause]) -> List[ExtractedClause]:
        """
        Re-classify clauses using RiskRules categories and weights
        """
        risk_classified = list()
        
        for clause in clauses:
            # Map to risk categories and calculate relevance
            risk_category, risk_confidence = self._classify_with_risk_categories(text = clause.text)
            
            # Update clause with risk classification
            clause.category                = risk_category
            clause.confidence              = risk_confidence
            
            risk_classified.append(clause)
        
        return risk_classified
    

    def _classify_with_risk_categories(self, text: str) -> Tuple[str, float]:
        """
        Classify text using RiskRules categories with contract-type weights
        """
        text_lower     = text.lower()
        
        # Keyword matching with risk categories
        keyword_scores = dict()
        
        for risk_category in self.risk_rules.CATEGORY_WEIGHTS.keys():
            # Get keywords from risk rules patterns
            keywords                      = self._get_keywords_for_risk_category(risk_category = risk_category)
            
            keyword_count                 = sum(1 for kw in keywords if kw in text_lower)
            base_score                    = (keyword_count / max(len(keywords), 1)) * 100
            
            # Apply contract-type specific weight
            weight                        = self.category_weights.get(risk_category, 1.0)
            keyword_scores[risk_category] = base_score * weight
        
        # Legal-BERT similarity with risk categories
        semantic_scores = dict()
        text_embedding  = self._get_legal_bert_embedding(text = text)
        text_tensor     = torch.tensor(text_embedding).unsqueeze(0)
        
        for risk_category, category_embedding in self.risk_category_embeddings.items():
            cat_tensor                     = torch.tensor(category_embedding).unsqueeze(0)
            similarity                     = torch.nn.functional.cosine_similarity(text_tensor, cat_tensor).item()
            semantic_scores[risk_category] = similarity * 100  # Scale to 0-100
        
        # Combine scores (60% semantic, 40% keyword)
        combined_scores = dict()
        
        for risk_category in self.risk_rules.CATEGORY_WEIGHTS.keys():
            combined                       = (semantic_scores.get(risk_category, 0) * 0.6 + keyword_scores.get(risk_category, 0) * 0.4)
            combined_scores[risk_category] = combined
        
        # Get best category
        best_category = max(combined_scores, key = combined_scores.get)

        # Normalize to 0-1
        confidence    = min(combined_scores[best_category] / 100, 1.0) 
        
        return best_category, confidence
    

    def _get_keywords_for_risk_category(self, risk_category: str) -> List[str]:
        """
        Get relevant keywords for a risk category from RiskRules patterns
        """
        # Map risk categories to relevant keywords from RiskRules
        keyword_map = {'restrictive_covenants' : ['non-compete', 'non-solicit', 'restrictive', 'covenant', 'competition', 'geographic', 'duration'],
                       'termination_rights'    : ['termination', 'notice', 'severance', 'dismissal', 'resignation', 'for cause', 'without cause'],
                       'penalties_liability'   : ['penalty', 'liquidated damages', 'liability', 'indemnification', 'hold harmless', 'damages'],
                       'compensation_benefits' : ['compensation', 'salary', 'benefits', 'bonus', 'commission', 'equity', 'insurance'],
                       'intellectual_property' : ['intellectual property', 'ip', 'copyright', 'patent', 'trademark', 'inventions'],
                       'confidentiality'       : ['confidential', 'proprietary', 'trade secret', 'non-disclosure'],
                       'liability_indemnity'   : ['liability', 'indemnification', 'hold harmless', 'defend', 'claims'],
                       'governing_law'         : ['governing law', 'jurisdiction', 'venue', 'dispute resolution'],
                       'payment_terms'         : ['payment', 'due', 'invoice', 'net 30', 'late payment'],
                       'warranties'            : ['warranty', 'representation', 'guarantee', 'disclaimer'],
                       'dispute_resolution'    : ['arbitration', 'mediation', 'dispute', 'litigation'],
                       'assignment_change'     : ['assignment', 'transfer', 'amendment', 'modification'],
                       'insurance'             : ['insurance', 'coverage', 'policy', 'insured'],
                       'force_majeure'         : ['force majeure', 'act of god', 'beyond control'],
                      }
        
        return keyword_map.get(risk_category, [])
    

    def _calculate_risk_scores(self, clauses: List[ExtractedClause]) -> List[ExtractedClause]:
        """
        Calculate risk scores for clauses based on RiskRules factors
        """
        for clause in clauses:
            risk_score        = self._calculate_single_clause_risk(clause = clause)
            clause.risk_score = risk_score
        
        return clauses
    

    def _calculate_single_clause_risk(self, clause: ExtractedClause) -> float:
        """
        Calculate risk score using RiskRules framework
        """
        base_score      = 0.0
        text_lower      = clause.text.lower()

        # Base risk from category weight (adjusted for contract type)
        category_weight = self.category_weights.get(clause.category, 1.0)
        base_score     += category_weight  

        # Add risk from CLAUSE_RISK_FACTORS (red flags)
        factor_config   = self.risk_rules.CLAUSE_RISK_FACTORS.get(clause.category)
        
        if factor_config:
            for red_flag, adjustment in factor_config["red_flags"].items():
                if red_flag in text_lower:
                    base_score += adjustment

        # Add risk from RISKY_PATTERNS (with actual scores)
        for pattern, score, description in self.risk_rules.RISKY_PATTERNS:
            if re.search(pattern, text_lower):
                base_score += score

        # Add risk from CRITICAL_KEYWORDS
        for keyword, risk_score in self.risk_rules.CRITICAL_KEYWORDS.items():
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                base_score += risk_score

        # Cap final score at 100
        return min(max(base_score, 0), 100)
    

    def _extract_risk_indicators(self, text: str) -> List[str]:
        """
        Extract risk indicators using RiskRules patterns
        """
        text_lower = text.lower()
        indicators = list()
        
        # Check critical risk patterns
        for pattern, score, description in self.risk_rules.RISKY_PATTERNS:
            if re.search(pattern, text_lower):
                indicators.append(description)
        
        # Check keyword risk indicators
        for indicator in self.risk_rules.CRITICAL_KEYWORDS.keys():
            if indicator in text_lower:
                indicators.append(indicator)
        
        for indicator in self.risk_rules.HIGH_RISK_KEYWORDS.keys():
            if indicator in text_lower:
                indicators.append(indicator)
        
        return indicators
    

    def _check_risk_patterns(self, text: str) -> float:
        """
        Check for high-risk patterns from RiskRules
        """
        text_lower   = text.lower()
        pattern_risk = 0.0
        
        # Check risky patterns
        for pattern, score, description in self.risk_rules.RISKY_PATTERNS:
            if re.search(pattern, text_lower):
                pattern_risk += score
        
        # Cap pattern risk
        return min(pattern_risk, 20)
    

    def _prioritize_risk_clauses(self, clauses: List[ExtractedClause], max_clauses: int) -> List[ExtractedClause]:
        """
        Prioritize clauses by risk score and contract-type relevance
        """
        # Sort by risk score (descending)
        clauses.sort(key = lambda x: x.risk_score, reverse = True)
        
        # Take top clauses
        return clauses[:max_clauses]
    

    def detect_missing_protections(self, extracted_clauses: List[ExtractedClause]) -> List[Dict]:
        """
        Detect missing critical protections based on contract type
        """
        missing   = list()
        checklist = self.risk_rules.PROTECTION_CHECKLIST
        
        for protection, config in checklist.items():
            if not self._has_protection(extracted_clauses, protection, config['categories']):
                missing.append({"protection"      : protection,
                                "importance"      : config['importance'],
                                "risk_if_missing" : config['risk_if_missing'],
                                "categories"      : config['categories'],
                              })
        
        log_info(f"Detected {len(missing)} missing protections")
        return missing
    

    def _has_protection(self, clauses: List[ExtractedClause], protection: str, categories: List[str]) -> bool:
        """
        Check if protection exists in extracted clauses
        """
        protection_patterns = {'for_cause_definition'     : ['for cause', 'cause defined', 'termination for cause'],
                               'severance_provision'      : ['severance', 'severance pay', 'termination benefits'],
                               'mutual_indemnification'   : ['mutual indemnification', 'both parties indemnify'],
                               'liability_cap'            : ['liability cap', 'limited liability', 'maximum liability'],
                               'prior_ip_exclusion'       : ['prior inventions', 'pre-existing ip', 'prior intellectual property'],
                               'confidentiality_duration' : ['confidentiality period', 'duration of confidentiality'],
                               'dispute_resolution'       : ['dispute resolution', 'arbitration', 'mediation'],
                               'change_control_process'   : ['change control', 'amendment process', 'modification procedure'],
                               'insurance_requirements'   : ['insurance requirements', 'maintain insurance'],
                               'force_majeure'            : ['force majeure', 'act of god'],
                              }
        
        patterns = protection_patterns.get(protection, [])
        
        for clause in clauses:
            if clause.category in categories:
                text_lower = clause.text.lower()
                if any(pattern in text_lower for pattern in patterns):
                    return True
        
        return False