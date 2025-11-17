# DEPENDENCIES
from enum import Enum
from typing import Dict
from typing import List
from typing import Tuple


class ContractType(Enum):
    EMPLOYMENT  = "employment"
    CONSULTING  = "consulting"
    NDA         = "nda"
    SOFTWARE    = "software"
    SERVICE     = "service"
    PARTNERSHIP = "partnership"
    LEASE       = "lease"
    PURCHASE    = "purchase"
    GENERAL     = "general"


class RiskRules:
    """
    Comprehensive risk scoring rules for broad contract coverage
    """
    CATEGORY_WEIGHTS          = {"restrictive_covenants" : 15,
                                 "termination_rights"    : 12,
                                 "penalties_liability"   : 14,
                                 "compensation_benefits" : 13,
                                 "intellectual_property" : 12,
                                 "confidentiality"       : 10,
                                 "liability_indemnity"   : 11,
                                 "governing_law"         : 8,
                                 "payment_terms"         : 10,
                                 "warranties"            : 9,
                                 "dispute_resolution"    : 7,
                                 "assignment_change"     : 6,
                                 "insurance"             : 5,
                                 "force_majeure"         : 4,
                                }
    
    # Contract-specific weight adjustments
    CONTRACT_TYPE_ADJUSTMENTS = {ContractType.EMPLOYMENT  : {"restrictive_covenants": 1.8, "compensation_benefits": 1.6, "termination_rights": 1.4, "confidentiality": 1.3},
                                 ContractType.SOFTWARE    : {"intellectual_property": 1.8, "penalties_liability": 1.5, "warranties": 1.4, "payment_terms": 1.3},
                                 ContractType.NDA         : {"confidentiality": 2.0, "penalties_liability": 1.6, "restrictive_covenants": 1.4},
                                 ContractType.CONSULTING  : {"compensation_benefits": 1.5, "termination_rights": 1.3, "liability_indemnity": 1.4},
                                 ContractType.LEASE       : {"payment_terms": 1.6, "termination_rights": 1.5, "liability_indemnity": 1.4},
                                 ContractType.PURCHASE    : {"warranties": 1.7, "payment_terms": 1.5, "liability_indemnity": 1.3},
                                 ContractType.PARTNERSHIP : {"governing_law": 1.6, "dispute_resolution": 1.5, "assignment_change": 1.4},
                                 ContractType.SERVICE     : {"payment_terms": 1.5, "warranties": 1.4, "termination_rights": 1.3},
                                } 
    
    CRITICAL_KEYWORDS         = {"non-compete"                : 25,
                                 "non-solicit"                : 23,
                                 "non-solicitation"           : 23,
                                 "forfeit"                    : 25,
                                 "liquidated damages"         : 24,
                                 "wage withholding"           : 25,
                                 "unlimited liability"        : 25,
                                 "joint and several"          : 23,
                                 "perpetual"                  : 22,
                                 "irrevocable"                : 20,
                                 "automatic renewal"          : 21,
                                 "assignment without consent" : 22,
                                 "sole discretion"            : 23,
                                }
    
    HIGH_RISK_KEYWORDS        = {"indemnify"             : 18,
                                 "indemnification"       : 18,
                                 "hold harmless"         : 17,
                                 "penalty"               : 18,
                                 "damages"               : 15,
                                 "breach"                : 15,
                                 "default"               : 14,
                                 "immediate termination" : 16,
                                 "without cause"         : 15,
                                 "at-will"               : 14,
                                 "waive"                 : 16,
                                 "release"               : 15,
                                 "confidential"          : 12,
                                 "proprietary"           : 12,
                                 "exclusive"             : 14,
                                 "non-refundable"        : 16,
                                }
    
    MEDIUM_RISK_KEYWORDS      = {"terminate"      : 7, 
                                 "termination"    : 7, 
                                 "assignment"     : 6, 
                                 "warranty"       : 8, 
                                 "representation" : 7, 
                                 "covenant"       : 8, 
                                 "jurisdiction"   : 6, 
                                 "governing law"  : 6, 
                                 "insurance"      : 5, 
                                 "force majeure"  : 4, 
                                 "amendment"      : 5,  
                                 "notice"         : 4,
                                }

    RISKY_PATTERNS            = [(r'\d+\s*(year|yr|month|mo)s?\s*(non-compete|non-solicit)', 20, "Long duration restrictive covenant"),
                                 (r'(entire|all|worldwide|global)\s*(industry|market|territory)', 18, "Overly broad geographic/industry scope"),
                                 (r'notice\s+period.*\d+\s*days.*employee.*\d+\s*days.*employer', 15, "Unequal notice periods"),
                                 (r'(may|can|shall)\s+(withhold|deduct|retain).*compensation', 22, "Wage withholding clause"),
                                 (r'(unlimited|no\s+limit|without\s+limitation).*liability', 25, "Unlimited liability exposure"),
                                 (r'(sole|absolute|unfettered)\s+discretion', 18, "One-sided discretionary power"),
                                 (r'penalty.*(?:equal\s+to|of|amount).*\$?\d+', 16, "Specific penalty amount"),
                                 (r'(automatically|immediately)\s+(renew|extend)', 12, "Auto-renewal clause"),
                                 (r'waive.*right.*arbitration', 20, "Arbitration rights waiver"),
                                 (r'(all|any).*intellectual\s+property.*created', 17, "Broad IP assignment"),
                                 (r'payment.*due.*upon.*signature', 14, "Payment due upon signature"),
                                 (r'no.*warranty.*(?:merchantability|fitness)', 15, "No warranty disclaimer"),
                                 (r'governing\s+law.*\b(?:delaware|nevada)\b', 8, "Specific governing law"),
                                ]

    CLAUSE_RISK_FACTORS       = {"non_compete"           : {"base_risk" : 70,
                                                            "red_flags" : {"same industry": 0, "direct competitor": -5, "entire industry": +20, "all industries": +25, "worldwide": +15, "global": +15, "specific city": -10, "specific state": -5},
                                                           },
                                 "termination"           : {"base_risk" : 50,
                                                            "red_flags" : {"without cause": +15, "immediate": +12, "at will": +10, "sole discretion": +18, "no notice": +20, "no reason": +15},
                                                           },
                                 "indemnification"       : {"base_risk" : 60,
                                                            "red_flags" : {"unlimited": +25, "joint and several": +20, "gross negligence": -10, "willful misconduct": -10, "third party": +8, "all claims": +15, "any claims": +15},
                                                           },
                                 "compensation"          : {"base_risk" : 30,
                                                            "red_flags" : {"to be determined": +20, "tbd": +20, "subject to review": +15, "discretionary": +18, "at employer's discretion": +22, "may withhold": +25, "can deduct": +20},
                                                           },
                                 "intellectual_property" : {"base_risk" : 55,
                                                            "red_flags" : {"all work product": +18, "anything created": +20, "during employment": +10, "after employment": +25, "including personal projects": +30, "whether or not related": +25},
                                                           },
                                 "confidentiality"       : {"base_risk" : 45,
                                                            "red_flags" : {"perpetual": +20, "indefinite": +18, "all information": +15, "including public information": +25},
                                                           },
                                 "payment"               : {"base_risk" : 35,
                                                            "red_flags" : {"net 90": +15, "net 120": +20, "upon completion": +10, "discretionary": +18},
                                                           },
                                 "warranty"              : {"base_risk" : 40,
                                                            "red_flags" : {"as is": +25, "no warranty": +20, "disclaims all": +22},
                                                           },
                                }
    
    INDUSTRY_BENCHMARKS       = {"non_compete_duration"     : {"tech"       : {"reasonable": 6, "standard": 12, "excessive": 24},
                                                               "finance"    : {"reasonable": 12, "standard": 18, "excessive": 36},
                                                               "healthcare" : {"reasonable": 12, "standard": 24, "excessive": 36},
                                                               "general"    : {"reasonable": 6, "standard": 12, "excessive": 24},
                                                              },
                                 "notice_period_days"       : {"executive"    : {"reasonable": 90, "standard": 60, "minimal": 30},
                                                               "senior"       : {"reasonable": 60, "standard": 30, "minimal": 14},
                                                               "professional" : {"reasonable": 30, "standard": 14, "minimal": 7},
                                                               "general"      : {"reasonable": 30, "standard": 14, "minimal": 7},
                                                              },
                                 "liability_cap_multiplier" : {"saas"        : {"generous": 24, "standard": 12, "restrictive": 3},
                                                               "consulting"  : {"generous": 3, "standard": 1, "restrictive": 0.5},
                                                               "general"     : {"generous": 12, "standard": 6, "restrictive": 1},
                                                              },
                                }

    PROTECTION_CHECKLIST      = {"for_cause_definition"     : {"importance": "critical", "risk_if_missing": 25, "categories": ["termination_rights"]},
                                 "severance_provision"      : {"importance": "high", "risk_if_missing": 18, "categories": ["termination_rights", "compensation_benefits"]},
                                 "mutual_indemnification"   : {"importance": "high", "risk_if_missing": 20, "categories": ["liability_indemnity"]},
                                 "liability_cap"            : {"importance": "critical", "risk_if_missing": 25, "categories": ["liability_indemnity", "penalties_liability"]},
                                 "prior_ip_exclusion"       : {"importance": "high", "risk_if_missing": 22, "categories": ["intellectual_property"]},
                                 "confidentiality_duration" : {"importance": "medium", "risk_if_missing": 12, "categories": ["confidentiality"]},
                                 "dispute_resolution"       : {"importance": "medium", "risk_if_missing": 15, "categories": ["dispute_resolution"]},
                                 "change_control_process"   : {"importance": "medium", "risk_if_missing": 10, "categories": ["assignment_change"]},
                                 "insurance_requirements"   : {"importance": "medium", "risk_if_missing": 12, "categories": ["insurance"]},
                                 "force_majeure"            : {"importance": "low", "risk_if_missing": 8, "categories": ["force_majeure"]},
                                }
                        
    RISK_THRESHOLDS           = {"critical" : 80,
                                 "high"     : 60,
                                 "medium"   : 40,
                                 "low"      : 20,
                                }
                            
    CATEGORY_DESCRIPTIONS     = {"restrictive_covenants"  : {"high"   : "Overly restrictive non-compete, non-solicit, or confidentiality terms that may significantly limit future opportunities",
                                                             "medium" : "Some restrictive terms present; review duration, geographic scope, and industry limitations",
                                                             "low"    : "Reasonable restrictive covenants appropriate for this role and industry standards",
                                                            },
                                 "termination_rights"     : {"high"   : "Unbalanced termination rights with immediate termination, 'at-will' clauses, or unequal notice periods favoring one party",
                                                             "medium" : "Moderately balanced termination provisions; review notice period requirements and severance terms",
                                                             "low"    : "Fair termination rights with reasonable notice periods and balanced severance provisions",
                                                            },
                                 "penalties_liability"    : {"high"   : "Excessive penalty clauses, unlimited liability exposure, or one-sided indemnification terms",
                                                             "medium" : "Some concerning liability terms; review indemnification scope, damage limitations, and warranty provisions",
                                                             "low"    : "Standard liability limitations, reasonable penalty provisions, and balanced indemnification terms",
                                                            },
                                 "compensation_benefits"  : {"high"   : "Compensation structure lacks clarity, contains vague terms, or has unfavorable payment conditions",
                                                             "medium" : "Compensation terms are generally clear but could benefit from more specific bonus structure and payment terms",
                                                             "low"    : "Clear and competitive compensation package with well-defined payment terms and bonus structure",
                                                            },
                                 "intellectual_property"  : {"high"   : "Overly broad IP assignment that may cover personal projects or lacks proper prior IP exclusion",
                                                             "medium" : "IP terms mostly clear but could benefit from stronger prior IP protection and clearer ownership terms",
                                                             "low"    : "Well-defined intellectual property ownership, clear usage rights, and proper prior IP exclusion",
                                                            },
                                 "confidentiality"        : {"high"   : "Overly broad confidentiality scope, perpetual duration, or insufficient protection exceptions",
                                                             "medium" : "Standard confidentiality terms with some areas that could be more precisely defined",
                                                             "low"    : "Reasonable confidentiality provisions with appropriate scope and duration",
                                                            },
                                 "liability_indemnity"    : {"high"   : "Unbalanced indemnification, unlimited liability exposure, or insufficient liability caps",
                                                             "medium" : "Moderate liability terms; review indemnification mutuality and liability limitations",
                                                             "low"    : "Balanced indemnification provisions with reasonable liability limitations",
                                                            },
                                 "governing_law"          : {"high"   : "Unfavorable jurisdiction selection, one-sided dispute resolution, or restrictive venue requirements",
                                                             "medium" : "Standard governing law terms with generally acceptable jurisdiction and dispute resolution",
                                                             "low"    : "Reasonable governing law and jurisdiction provisions favorable to both parties",
                                                            },
                                 "payment_terms"          : {"high"   : "Unfavorable payment terms, extended payment periods, or unclear payment conditions",
                                                             "medium" : "Standard payment terms with some areas that could be improved for cash flow",
                                                             "low"    : "Favorable payment terms with reasonable payment periods and clear conditions",
                                                            },
                                 "warranties"             : {"high"   : "Overly broad warranty disclaimers, insufficient product guarantees, or one-sided warranty terms",
                                                             "medium" : "Standard warranty provisions with typical product/service guarantees",
                                                             "low"    : "Comprehensive warranty coverage with reasonable limitations and clear guarantees",
                                                            },
                                 "dispute_resolution"     : {"high"   : "Unfavorable dispute resolution process, restrictive arbitration clauses, or one-sided legal fee allocation",
                                                             "medium" : "Standard dispute resolution terms with generally fair arbitration or litigation process",
                                                             "low"    : "Reasonable dispute resolution process with fair arbitration and cost allocation",
                                                            },
                                 "assignment_change"      : {"high"   : "Restrictive assignment clauses, one-sided change control, or unfavorable amendment procedures",
                                                             "medium" : "Standard assignment and change control terms with reasonable flexibility",
                                                             "low"    : "Reasonable assignment rights and change control processes favorable to both parties",
                                                            },
                                 "insurance"              : {"high"   : "Insufficient insurance requirements, unclear coverage terms, or inadequate policy specifications",
                                                             "medium" : "Standard insurance requirements with typical coverage expectations",
                                                             "low"    : "Comprehensive insurance requirements with clear coverage specifications",
                                                            },
                                 "force_majeure"          : {"high"   : "Overly narrow force majeure definition, insufficient relief provisions, or one-sided termination rights",
                                                             "medium" : "Standard force majeure clause with typical relief provisions",
                                                             "low"    : "Comprehensive force majeure protection with reasonable relief and termination rights",
                                                            },
                                }
        
    PROTECTION_NAME_MAP       = {"for_cause_definition"     : "For Cause Definition",
                                 "severance_proportion"     : "Severance Provision",
                                 "mutual_indemnification"   : "Mutual Indemnification",
                                 "liability_cap"            : "Liability Cap",
                                 "prior_ip_exclusion"       : "Prior IP Exclusion",
                                 "confidentiality_duration" : "Confidentiality Duration Limit",
                                 "dispute_resolution"       : "Dispute Resolution Process",
                                 "change_control_process"   : "Change Control Process",
                                 "insurance_requirements"   : "Insurance Requirements",
                                 "force_majeure"            : "Force Majeure Protection",
                                }

    @classmethod
    def get_adjusted_weights(cls, contract_type: ContractType) -> Dict[str, float]:
        """
        Get category weights adjusted for contract type
        """
        base_weights = cls.CATEGORY_WEIGHTS.copy()
        adjustments  = cls.CONTRACT_TYPE_ADJUSTMENTS.get(contract_type, {})
        
        adjusted    = dict()

        for category, weight in base_weights.items():
            multiplier         = adjustments.get(category, 1.0)
            adjusted[category] = weight * multiplier
        
        # Normalize to sum to 100
        total            = sum(adjusted.values())

        adjusted_weights = {k: (v / total) * 100 for k, v in adjusted.items()}
        
        return adjusted_weights
    

    @classmethod
    def get_category_description(cls, category: str, score: int) -> str:
        """
        Get meaningful description for a category based on score
        """
        if category not in cls.CATEGORY_DESCRIPTIONS:
            return "Review recommended based on risk score"
        
        if (score >= 70):
            risk_level = "high"

        elif (score >= 40):
            risk_level = "medium"

        else:
            risk_level = "low"
        
        category_description = cls.CATEGORY_DESCRIPTIONS[category][risk_level]
        
        return category_description


    @classmethod
    def get_protection_display_name(cls, protection_id: str) -> str:
        """
        Get the display name for a protection ID: Uses PROTECTION_NAME_MAP for known IDs, otherwise formats the ID
        """
        return cls.PROTECTION_NAME_MAP.get(protection_id, protection_id.replace("_", " ").title())
