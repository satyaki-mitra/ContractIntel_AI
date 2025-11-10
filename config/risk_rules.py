from typing import Dict, List, Tuple
from enum import Enum

class ContractType(Enum):
    EMPLOYMENT = "employment"
    CONSULTING = "consulting"
    NDA = "nda"
    SOFTWARE = "software"
    SERVICE = "service"
    PARTNERSHIP = "partnership"
    LEASE = "lease"
    PURCHASE = "purchase"
    GENERAL = "general"


class RiskRules:
    """Comprehensive risk scoring rules without ML training"""
    
    # =========================================================================
    # CATEGORY WEIGHTS (Base weights, adjusted by contract type)
    # =========================================================================
    
    CATEGORY_WEIGHTS = {
        "restrictive_covenants": 25,
        "termination_rights": 20,
        "penalties_liability": 20,
        "compensation_benefits": 15,
        "intellectual_property": 20
    }
    
    # Contract-specific weight adjustments
    CONTRACT_TYPE_ADJUSTMENTS = {
        ContractType.EMPLOYMENT: {
            "restrictive_covenants": 1.3,  # Higher importance
            "compensation_benefits": 1.4,
            "termination_rights": 1.2
        },
        ContractType.SOFTWARE: {
            "intellectual_property": 1.5,
            "penalties_liability": 1.3
        },
        ContractType.NDA: {
            "restrictive_covenants": 1.8,
            "penalties_liability": 1.2
        }
    }
    
    # =========================================================================
    # KEYWORD SEVERITY SCORING (Multi-tier system)
    # =========================================================================
    
    # Critical keywords (Tier 1: 20-25 points each)
    CRITICAL_KEYWORDS = {
        "non-compete": 25,
        "non-solicit": 23,
        "non-solicitation": 23,
        "forfeit": 25,
        "liquidated damages": 24,
        "wage withholding": 25,
        "unlimited liability": 25,
        "joint and several": 23,
        "perpetual": 22,
        "irrevocable": 20
    }
    
    # High-risk keywords (Tier 2: 12-18 points)
    HIGH_RISK_KEYWORDS = {
        "indemnify": 18,
        "indemnification": 18,
        "hold harmless": 17,
        "penalty": 18,
        "damages": 15,
        "breach": 15,
        "default": 14,
        "immediate termination": 16,
        "without cause": 15,
        "sole discretion": 17,
        "at-will": 14,
        "waive": 16,
        "release": 15
    }
    
    # Medium-risk keywords (Tier 3: 6-10 points)
    MEDIUM_RISK_KEYWORDS = {
        "confidential": 8,
        "proprietary": 8,
        "trade secret": 10,
        "terminate": 7,
        "termination": 7,
        "assignment": 6,
        "exclusive": 9,
        "warranty": 8,
        "representation": 7,
        "covenant": 8,
        "jurisdiction": 6,
        "governing law": 6
    }
    
    # =========================================================================
    # STRUCTURAL PATTERN ANALYSIS (Pattern-based risk detection)
    # =========================================================================
    
    RISKY_PATTERNS = [
        # Pattern: (regex, risk_points, description)
        (r'\d+\s*(year|yr|month|mo)s?\s*(non-compete|non-solicit)', 20, 
         "Long duration restrictive covenant"),
        
        (r'(entire|all|worldwide|global)\s*(industry|market|territory)', 18,
         "Overly broad geographic/industry scope"),
        
        (r'notice\s+period.*\d+\s*days.*employee.*\d+\s*days.*employer', 15,
         "Unequal notice periods"),
        
        (r'(may|can|shall)\s+(withhold|deduct|retain).*compensation', 22,
         "Wage withholding clause"),
        
        (r'(unlimited|no\s+limit|without\s+limitation).*liability', 25,
         "Unlimited liability exposure"),
        
        (r'(sole|absolute|unfettered)\s+discretion', 18,
         "One-sided discretionary power"),
        
        (r'penalty.*(?:equal\s+to|of|amount).*\$?\d+', 16,
         "Specific penalty amount"),
        
        (r'(automatically|immediately)\s+(renew|extend)', 12,
         "Auto-renewal clause"),
        
        (r'waive.*right.*arbitration', 20,
         "Arbitration rights waiver"),
        
        (r'(all|any).*intellectual\s+property.*created', 17,
         "Broad IP assignment"),
    ]
    
    # =========================================================================
    # CLAUSE-LEVEL RISK FACTORS (Detailed clause analysis)
    # =========================================================================
    
    CLAUSE_RISK_FACTORS = {
        "non-compete": {
            "base_risk": 70,
            "duration_check": {
                # months: risk_adjustment
                0: -20,   # No restriction
                6: -10,   # 6 months reasonable
                12: 0,    # 1 year standard
                18: +10,  # 18 months concerning
                24: +20,  # 2 years high risk
                36: +30,  # 3+ years critical
            },
            "scope_keywords": {
                "same industry": 0,
                "direct competitor": -5,
                "entire industry": +20,
                "all industries": +25,
                "worldwide": +15,
                "global": +15,
                "specific city": -10,
                "specific state": -5,
            }
        },
        
        "termination": {
            "base_risk": 50,
            "red_flags": {
                "without cause": +15,
                "immediate": +12,
                "at will": +10,
                "sole discretion": +18,
                "no notice": +20,
                "no reason": +15
            },
            "notice_period_imbalance": {
                # If employee notice > employer notice
                "ratio_2x": +10,
                "ratio_3x": +18,
                "ratio_4x": +25
            }
        },
        
        "indemnification": {
            "base_risk": 60,
            "red_flags": {
                "unlimited": +25,
                "joint and several": +20,
                "gross negligence": -10,  # Good - limits scope
                "willful misconduct": -10,  # Good - limits scope
                "third party": +8,
                "all claims": +15,
                "any claims": +15
            },
            "mutual_vs_onesided": {
                "mutual": -15,  # Good - balanced
                "one_sided": +20  # Bad - unbalanced
            }
        },
        
        "compensation": {
            "base_risk": 30,
            "red_flags": {
                "to be determined": +20,
                "tbd": +20,
                "subject to review": +15,
                "discretionary": +18,
                "at employer's discretion": +22,
                "may withhold": +25,
                "can deduct": +20
            },
            "clarity_bonus": {
                "specific_amount": -15,
                "payment_schedule": -10,
                "bonus_criteria": -8
            }
        },
        
        "intellectual_property": {
            "base_risk": 55,
            "red_flags": {
                "all work product": +18,
                "anything created": +20,
                "during employment": +10,
                "after employment": +25,
                "including personal projects": +30,
                "whether or not related": +25
            },
            "protections": {
                "prior ip excluded": -15,
                "personal projects excluded": -20,
                "work for hire limited": -10
            }
        },
        
        "liability": {
            "base_risk": 65,
            "red_flags": {
                "unlimited": +30,
                "consequential damages": +15,
                "indirect damages": +12,
                "punitive damages": +18,
                "no cap": +25
            },
            "protections": {
                "liability cap": -20,
                "mutual cap": -15,
                "limited to fees paid": -18
            }
        },
        
        "confidentiality": {
            "base_risk": 45,
            "red_flags": {
                "perpetual": +20,
                "forever": +20,
                "indefinite": +18,
                "all information": +15,
                "any information": +15
            },
            "reasonable_terms": {
                "3 years": -5,
                "5 years": 0,
                "7 years": +5,
                "marked confidential": -8,
                "reasonably necessary": -10
            }
        }
    }
    
    # =========================================================================
    # INDUSTRY BENCHMARKS (Standard market terms)
    # =========================================================================
    
    INDUSTRY_BENCHMARKS = {
        "non_compete_duration": {
            "tech": {"reasonable": 6, "standard": 12, "excessive": 24},
            "finance": {"reasonable": 12, "standard": 18, "excessive": 36},
            "healthcare": {"reasonable": 12, "standard": 24, "excessive": 36},
            "general": {"reasonable": 6, "standard": 12, "excessive": 24}
        },
        
        "notice_period_days": {
            "executive": {"reasonable": 90, "standard": 60, "minimal": 30},
            "senior": {"reasonable": 60, "standard": 30, "minimal": 14},
            "professional": {"reasonable": 30, "standard": 14, "minimal": 7},
            "general": {"reasonable": 30, "standard": 14, "minimal": 7}
        },
        
        "liability_cap_multiplier": {
            "saas": {"generous": 24, "standard": 12, "restrictive": 3},
            "consulting": {"generous": 3, "standard": 1, "restrictive": 0.5},
            "general": {"generous": 12, "standard": 6, "restrictive": 1}
        },
        
        "ip_assignment_scope": {
            "tech": "work_product_only",  # Standard
            "creative": "commissioned_work_only",  # Standard
            "consulting": "deliverables_only",  # Standard
            "general": "work_for_hire"  # Standard
        }
    }
    
    # =========================================================================
    # MISSING PROTECTIONS (Scored by importance)
    # =========================================================================
    
    PROTECTION_CHECKLIST = {
        "for_cause_definition": {
            "importance": "critical",
            "risk_if_missing": 25,
            "categories": ["termination"]
        },
        "severance_provision": {
            "importance": "high",
            "risk_if_missing": 18,
            "categories": ["termination", "compensation"]
        },
        "mutual_indemnification": {
            "importance": "high",
            "risk_if_missing": 20,
            "categories": ["liability"]
        },
        "liability_cap": {
            "importance": "critical",
            "risk_if_missing": 25,
            "categories": ["liability"]
        },
        "prior_ip_exclusion": {
            "importance": "high",
            "risk_if_missing": 22,
            "categories": ["intellectual_property"]
        },
        "confidentiality_duration": {
            "importance": "medium",
            "risk_if_missing": 12,
            "categories": ["confidentiality"]
        },
        "dispute_resolution": {
            "importance": "medium",
            "risk_if_missing": 15,
            "categories": ["general"]
        },
        "change_control_process": {
            "importance": "medium",
            "risk_if_missing": 10,
            "categories": ["general"]
        }
    }
    
    # =========================================================================
    # RISK LEVEL THRESHOLDS
    # =========================================================================
    
    RISK_THRESHOLDS = {
        "critical": 80,
        "high": 60,
        "medium": 40,
        "low": 20
    }
    
    @classmethod
    def get_adjusted_weights(cls, contract_type: ContractType) -> Dict[str, float]:
        """Get category weights adjusted for contract type"""
        base_weights = cls.CATEGORY_WEIGHTS.copy()
        adjustments = cls.CONTRACT_TYPE_ADJUSTMENTS.get(contract_type, {})
        
        adjusted = {}
        for category, weight in base_weights.items():
            multiplier = adjustments.get(category, 1.0)
            adjusted[category] = weight * multiplier
        
        # Normalize to sum to 100
        total = sum(adjusted.values())
        return {k: (v / total) * 100 for k, v in adjusted.items()}

