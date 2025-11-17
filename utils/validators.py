# DEPENDENCIES
import re
import os
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path


class ContractValidator:
    """
    Validate if document is a legal contract
    """
    # File constraints
    MIN_CONTRACT_LENGTH = 500    
    MAX_CONTRACT_LENGTH = 500000  # 500KB text
    
    # Strong indicators of legal contracts (keyword: weight)
    STRONG_INDICATORS   = {'agreement'             : 3, 
                           'contract'              : 3, 
                           'party'                 : 2, 
                           'parties'               : 2, 
                           'whereas'               : 5, 
                           'hereinafter'           : 5, 
                           'witnesseth'            : 5, 
                           'indemnification'       : 4, 
                           'liability'             : 3, 
                           'confidentiality'       : 3, 
                           'termination'           : 3, 
                           'governing law'         : 4, 
                           'jurisdiction'          : 3, 
                           'warranty'              : 3, 
                           'representation'        : 3, 
                           'covenant'              : 4, 
                           'clause'                : 3, 
                           'section'               : 2,
                           'article'               : 2, 
                           'hereby'                : 3, 
                           'undersigned'           : 4, 
                           'executed'              : 3,
                           'consideration'         : 4, 
                           'effective date'        : 3, 
                           'in witness whereof'    : 5, 
                           'binding'               : 3, 
                           'enforceable'           : 3, 
                           'obligations'           : 2,
                           'employment'            : 3, 
                           'employee'              : 2, 
                           'employer'              : 2,
                           'probation'             : 3, 
                           'salary'                : 2, 
                           'compensation'          : 3,
                           'non-compete'           : 4, 
                           'non-solicit'           : 4,
                           'remuneration'          : 3, 
                           'indemnity'             : 3, 
                           'intellectual property' : 4,
                           'confidential'          : 2, 
                           'proprietary'           : 2, 
                           'post-termination'      : 3,
                           'agrees to'             : 2, 
                           'shall not'             : 2, 
                           'agrees and accepts'    : 3,
                           'subject to'            : 1, 
                           'in accordance with'    : 2,
                          }
    
    # Anti-patterns (things that indicate NOT a contract)
    ANTI_PATTERNS       = {'case law'           : 5, 
                           'plaintiff'          : 5, 
                           'defendant'          : 5, 
                           'supreme court'      : 5, 
                           'appellate court'    : 5, 
                           'court held'         : 5, 
                           'legal opinion'      : 4, 
                           'court of appeals'   : 5, 
                           'trial court'        : 5, 
                           'article written by' : 4, 
                           'blog post'          : 5, 
                           'this article'       : 3,
                           'author:'            : 3, 
                           'published in'       : 3, 
                           'journal of'         : 3, 
                           'abstract:'          : 4, 
                           'introduction:'      : 3, 
                           'conclusion:'        : 3, 
                           'table of contents'  : 4, 
                           'bibliography'       : 4,
                           'references:'        : 3, 
                           'chapter'            : 2, 
                           'section i.'         : 2, 
                           'section ii.'        : 2,
                          }
    

    @staticmethod
    def is_valid_contract(text: str, min_length: int = None) -> Tuple[bool, str, str]:
        """
        Comprehensive contract validation with relaxed thresholds
        
        Arguments:
        ----------
            text       { str } : Document text to validate
            
            min_length { int } : Minimum length override (optional)
        
        Returns:
        --------
               { tuple }       : (is_valid, validation_type, message) tuple
        """
        min_length = min_length or ContractValidator.MIN_CONTRACT_LENGTH
        text_lower = text.lower().strip()
        
        # Length Validation
        if (len(text_lower) < min_length):
            return (False, "too_short", f"Text too short ({len(text_lower)} chars, minimum {min_length}). This is likely a snippet, not a full contract.")
        
        if (len(text_lower) > ContractValidator.MAX_CONTRACT_LENGTH):
            return (False, "too_long", f"Text too long ({len(text_lower)} chars, maximum {ContractValidator.MAX_CONTRACT_LENGTH}). This may be a contract bundle or combined document.")
        
        # Anti-pattern Check (Prevent False Positives)
        anti_score          = 0
        found_anti_patterns = list()
        
        for pattern, weight in ContractValidator.ANTI_PATTERNS.items():
            if pattern in text_lower:
                anti_score += weight
                found_anti_patterns.append(pattern)
        
        # More strict anti-pattern check
        if (anti_score >= 10):  # Reduced from 15
            return (False, "not_contract", f"The provided document does not appear to be a legal contract. Please upload a valid contract for analysis.")
        
        # Positive Indicator Scoring
        score            = 0
        found_indicators = list()
        
        for indicator, weight in ContractValidator.STRONG_INDICATORS.items():
            if indicator in text_lower:
                score += weight
                found_indicators.append(indicator)
        
        # Structural Pattern Analysis
        structural_score = ContractValidator._check_structural_patterns(text = text_lower)
        score           += structural_score
        
        # Signature Block Check
        has_signature_block = ContractValidator._has_signature_block(text = text_lower)
        if has_signature_block:
            score += 5
            found_indicators.append("signature block")
        
        # Effective Date Check
        has_effective_date = ContractValidator._has_effective_date(text = text)
        if has_effective_date:
            score += 3
            found_indicators.append("effective date")
        
        # Party Identification Check
        has_parties = ContractValidator._has_party_identification(text = text)
        if has_parties:
            score += 4
            found_indicators.append("party identification")
        
        # Validation Thresholds 
        if (score >= 50): 
            return (True, "high_confidence", f"Strong contract indicators detected (score: {score}). This is highly likely a legal contract.")
        
        elif (score >= 40):  # Reduced from 15 (now accepts lower confidence)
            return (True, "medium_confidence", f"Contract indicators present (score: {score}). This appears to be a contract.")
        
        elif (score >= 25):  
            return (True, "low_confidence", f"Some contract indicators present (score: {score}). Proceeding with analysis.")
        
        else:
            return (False, "not_contract", f"The provided document does not appear to be a legal contract. Please upload a valid contract for analysis.")
    

    @staticmethod
    def _check_structural_patterns(text: str) -> int:
        """
        Check for structural patterns unique to contracts
        """
        score    = 0
        patterns = [(r'in\s+consideration\s+of', 3),
                    (r'now,?\s+therefore', 3),
                    (r'agree\s+as\s+follows', 3),
                    (r'in\s+witness\s+whereof', 4),
                    (r'this\s+agreement.*(?:made|entered)', 3),
                    (r'between.*and.*(?:collectively|hereinafter)', 3),
                    (r'effective\s+as\s+of', 2),
                    (r'signed.*presence\s+of', 2),
                    (r'intending\s+to\s+be\s+legally\s+bound', 4),
                    (r'mutually\s+agree', 2),
                    (r'terms\s+and\s+conditions', 2),
                   ]
        
        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        return score
    

    @staticmethod
    def _has_signature_block(text: str) -> bool:
        """
        Check for signature block patterns
        """
        signature_patterns = [r'signature:?\s*_+',
                              r'signed:?\s*_+',
                              r'by:?\s*_+',
                              r'name:?\s*_+.*title:?\s*_+',
                              r'\[signature\]',
                              r'\[seal\]',
                              r'authorized\s+signatory',
                              r'in\s+witness\s+whereof.*executed',
                             ]
        
        return any(re.search(p, text, re.IGNORECASE) for p in signature_patterns)
    

    @staticmethod
    def _has_effective_date(text: str) -> bool:
        """
        Check for effective date patterns
        """
        date_patterns = [r'effective\s+(?:date|as\s+of)',
                         r'dated\s+as\s+of',
                         r'this\s+\d+(?:st|nd|rd|th)?\s+day\s+of',
                         r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
                         r'commencement\s+date',
                         r'execution\s+date',
                        ]
        
        return any(re.search(p, text, re.IGNORECASE) for p in date_patterns)
    

    @staticmethod
    def _has_party_identification(text: str) -> bool:
        """
        Check if parties are clearly identified
        """
        party_patterns = [r'between.*and.*\(.*".*"\)',
                          r'party\s+[a-z]\s*[:\-]',
                          r'(?:the\s+)?(?:employer|employee|consultant|contractor|client|vendor|landlord|tenant|buyer|seller)',
                          r'hereinafter\s+referred\s+to\s+as',
                          r'\("(?:the\s+)?(?:company|employee|consultant)"\)',
                          r'first\s+party.*second\s+party',
                         ]
        
        return any(re.search(p, text, re.IGNORECASE) for p in party_patterns)
    

    @staticmethod
    def validate_file_integrity(file_path: str) -> Tuple[bool, str]:
        """
        Validate file isn't corrupted and is readable
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "File does not exist"
            
            file_size = file_path.stat().st_size
            
            if (file_size == 0):
                return False, "File is empty (0 bytes)"
            
            if (file_size < 1024):  
                return (False, f"File suspiciously small ({file_size} bytes)")
            
            with open(file_path, 'rb') as f:
                first_kb = f.read(1024)
                if (b'\x00' * 10 in first_kb):
                    return (False, "File appears corrupted (contains null bytes)")
            
            return (True, "File integrity OK")
            
        except PermissionError:
            return (False, "Permission denied - cannot read file")
        except Exception as e:
            return (False, f"File integrity check failed: {repr(e)}")
    

    @staticmethod
    def get_validation_report(text: str) -> Dict[str, any]:
        """
        Get detailed validation report with scores and findings
        """
        is_valid, validation_type, message = ContractValidator.is_valid_contract(text = text)
        
        text_lower                         = text.lower()
        
        # Calculate individual scores
        indicator_score                    = sum(weight for indicator, weight in ContractValidator.STRONG_INDICATORS.items() if indicator in text_lower)
        anti_score                         = sum(weight for pattern, weight in ContractValidator.ANTI_PATTERNS.items() if pattern in text_lower)
        structural_score                   = ContractValidator._check_structural_patterns(text = text_lower)
        
        # Collect found indicators
        found_indicators                   = [indicator for indicator in ContractValidator.STRONG_INDICATORS.keys() if indicator in text_lower]
        found_anti_patterns                = [pattern for pattern in ContractValidator.ANTI_PATTERNS.keys() if pattern in text_lower]
        
        return {"is_valid"            : is_valid,
                "validation_type"     : validation_type,
                "message"             : message,
                "scores"              : {"total"         : indicator_score + structural_score,
                                         "indicators"    : indicator_score,
                                         "structural"    : structural_score,
                                         "anti_patterns" : anti_score,
                                        },
                "features"            : {"has_signature_block"      : ContractValidator._has_signature_block(text = text_lower),
                                         "has_effective_date"       : ContractValidator._has_effective_date(text = text),
                                         "has_party_identification" : ContractValidator._has_party_identification(text = text),
                                        },
                "found_indicators"    : found_indicators,
                "found_anti_patterns" : found_anti_patterns,
                "text_statistics"     : {"length"     : len(text),
                                         "word_count" : len(text.split()),
                                         "line_count" : len(text.split('\n')),
                                        }
               }