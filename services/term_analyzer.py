# DEPENDENCIES
import re
import sys
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.risk_rules import RiskRules
from config.risk_rules import ContractType
from utils.logger import ContractAnalyzerLogger
from services.data_models import ExtractedClause
from services.data_models import UnfavorableTerm


class TermAnalyzer:
    """
    Detect unfavorable and one-sided terms in contracts using RiskRules framework and integrated with comprehensive risk analysis system
    """
    def __init__(self, contract_type: ContractType = ContractType.GENERAL):
        """
        Initialize term analyzer with contract-type specific risk rules
        
        Arguments:
        ----------
            contract_type { ContractType } : Contract type for risk rule adjustments
        """
        self.contract_type    = contract_type
        self.risk_rules       = RiskRules()
        self.logger           = ContractAnalyzerLogger.get_logger()
        
        # Contract-type specific weights
        self.category_weights = self.risk_rules.get_adjusted_weights(contract_type)
        
        log_info("TermAnalyzer initialized", 
                 contract_type    = contract_type.value,
                 category_weights = self.category_weights,
                )

    
    def _map_to_risk_category(self, clause_category: str) -> str:
        """
        Map clause category to risk category for proper risk scoring for ensureing unfavorable terms are correctly attributed to risk categories
        for score calculation
        """
        # Clause categories → Risk categories
        mapping                          = {"non_compete"           : "restrictive_covenants",
                                            "confidentiality"       : "restrictive_covenants",
                                            "termination"           : "termination_rights",
                                            "indemnification"       : "liability_indemnity",
                                            "liability"             : "penalties_liability",
                                            "compensation"          : "compensation_benefits",
                                            "intellectual_property" : "intellectual_property",
                                            "warranty"              : "warranties",
                                            "dispute_resolution"    : "dispute_resolution",
                                            "assignment"            : "assignment_change",
                                            "amendment"             : "assignment_change",
                                            "insurance"             : "insurance",
                                            "force_majeure"         : "force_majeure",
                                            "general"               : "general",
                                            "payment"               : "payment_terms",
                                            "governing_law"         : "governing_law",
                                           }

        risk_category_by_clause_category = mapping.get(clause_category, clause_category)
        
        return risk_category_by_clause_category
    

    @ContractAnalyzerLogger.log_execution_time("analyze_unfavorable_terms")
    def analyze_unfavorable_terms(self, contract_text: str, clauses: List[ExtractedClause], contract_type: Optional[ContractType] = None) -> List[UnfavorableTerm]:
        """
        Detect all unfavorable terms using RiskRules framework
        
        Arguments:
        ----------
            contract_text { str }          : Full contract text
            
            clauses       { list }         : Extracted clauses
            
            contract_type { ContractType } : Override contract type
        
        Returns:
        --------
                      { list }             : List of UnfavorableTerm objects
        """
        # Update contract type if provided
        if contract_type:
            self.contract_type    = contract_type
            self.category_weights = self.risk_rules.get_adjusted_weights(contract_type)
        
        log_info("Starting unfavorable terms analysis",
                 text_length    = len(contract_text),
                 num_clauses    = len(clauses),
                 contract_type  = self.contract_type.value,
                )
        
        unfavorable_terms = list()
        
        # Clause-level analysis using RiskRules patterns
        for clause in clauses:
            terms = self._analyze_clause_with_risk_rules(clause = clause)
            unfavorable_terms.extend(terms)
        
        # Cross-clause analysis for systemic issues
        cross_clause_terms = self._analyze_cross_clause_issues(text    = contract_text, 
                                                               clauses = clauses,
                                                              )
        unfavorable_terms.extend(cross_clause_terms)
        
        # PHASE 3: Missing protections analysis
        missing_protections = self._analyze_missing_protections(clauses = clauses)
        unfavorable_terms.extend(missing_protections)
        
        # PHASE 4: Industry benchmark analysis
        benchmark_issues = self._analyze_against_benchmarks(clauses = clauses)
        unfavorable_terms.extend(benchmark_issues)
        
        # Deduplicate and prioritize by risk
        final_terms = self._deduplicate_and_prioritize(terms = unfavorable_terms)
        
        log_info("Unfavorable terms analysis complete",
                 total_found = len(final_terms),
                 critical    = sum(1 for t in final_terms if (t.severity == "critical")),
                 high        = sum(1 for t in final_terms if (t.severity == "high")))
        
        return final_terms
    

    def _analyze_clause_with_risk_rules(self, clause: ExtractedClause) -> List[UnfavorableTerm]:
        """
        Analyze clause using comprehensive RiskRules framework
        """
        terms      = list()
        text_lower = clause.text.lower()
        
        # Map clause category to risk category for consistency
        risk_category = self._map_to_risk_category(clause_category = clause.category)
        
        # Risky Patterns Analysis from RiskRules
        for pattern, risk_score, description in self.risk_rules.RISKY_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                severity = self._score_to_severity(risk_score)
                
                terms.append(UnfavorableTerm(term             = description,
                                             category         = risk_category,
                                             severity         = severity,
                                             explanation      = self._generate_pattern_explanation(description, match.group()),
                                             risk_score       = risk_score,
                                             clause_reference = clause.reference,
                                             suggested_fix    = self._generate_pattern_fix(description, clause.category),
                                             contract_type    = self.contract_type.value,
                                             specific_text    = match.group(),
                                             legal_basis      = self._get_legal_basis(description),
                                            )
                            )
        
        # Critical Keyword Analysis from RiskRules
        for keyword, risk_score in self.risk_rules.CRITICAL_KEYWORDS.items():
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                severity = self._score_to_severity(risk_score)
                
                terms.append(UnfavorableTerm(term             = f"Critical Risk: {keyword.title()}",
                                             category         = risk_category,
                                             severity         = severity,
                                             explanation      = self._generate_keyword_explanation(keyword, clause.category),
                                             risk_score       = risk_score,
                                             clause_reference = clause.reference,
                                             suggested_fix    = self._generate_keyword_fix(keyword, clause.category),
                                             contract_type    = self.contract_type.value,
                                             specific_text    = keyword,
                                             legal_basis      = self._get_legal_basis(keyword),
                                            )
                            )
                                
        # High Risk Keyword Analysis 
        for keyword, risk_score in self.risk_rules.HIGH_RISK_KEYWORDS.items():
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                severity = self._score_to_severity(risk_score)
                
                terms.append(UnfavorableTerm(term             = f"High Risk: {keyword.title()}",
                                             category         = risk_category, 
                                             severity         = severity,
                                             explanation      = self._generate_keyword_explanation(keyword, clause.category),
                                             risk_score       = risk_score,
                                             clause_reference = clause.reference,
                                             suggested_fix    = self._generate_keyword_fix(keyword, clause.category),
                                             contract_type    = self.contract_type.value,
                                             specific_text    = keyword,
                                             legal_basis      = self._get_legal_basis(keyword),
                                            )
                            )
        
        # Clause-specific Risk Factors From RiskRules.CLAUSE_RISK_FACTORS
        clause_risk_analysis = self._analyze_clause_risk_factors(clause)
        terms.extend(clause_risk_analysis)
        
        return terms
    

    def _analyze_clause_risk_factors(self, clause: ExtractedClause) -> List[UnfavorableTerm]:
        """
        Analyze clause using CLAUSE_RISK_FACTORS from RiskRules
        """
        terms            = list()
        
        # Map clause categories to risk factors
        category_mapping = {'non_compete'           : 'restrictive_covenants',
                            'termination'           : 'termination_rights', 
                            'indemnification'       : 'liability_indemnity',
                            'compensation'          : 'compensation_benefits',
                            'intellectual_property' : 'intellectual_property',
                            'confidentiality'       : 'confidentiality',
                            'liability'             : 'penalties_liability',
                            'warranty'              : 'warranties',
                            'dispute_resolution'    : 'dispute_resolution',
                            'assignment'            : 'assignment_change',
                            'insurance'             : 'insurance',
                            'force_majeure'         : 'force_majeure',
                           }
        
        risk_factors_key = category_mapping.get(clause.category)
        if not risk_factors_key or risk_factors_key not in self.risk_rules.CLAUSE_RISK_FACTORS:
            return terms
        
        risk_factors = self.risk_rules.CLAUSE_RISK_FACTORS[risk_factors_key]
        text_lower   = clause.text.lower()
        
        # Map clause category to risk category for consistency
        risk_category = self._map_to_risk_category(clause_category = clause.category)

        # Check for red flags in this clause
        for red_flag, risk_adjustment in risk_factors["red_flags"].items():
            if (red_flag in text_lower):
                base_risk    = risk_factors["base_risk"]
                total_risk   = base_risk + risk_adjustment
                severity     = self._score_to_severity(total_risk)
                
                terms.append(UnfavorableTerm(term             = f"Risk Factor: {red_flag.replace('_', ' ').title()}",
                                             category         = risk_category,
                                             severity         = severity,
                                             explanation      = f"Base risk {base_risk} + {risk_adjustment} for '{red_flag}'. {self._get_risk_factor_explanation(risk_factors_key, red_flag)}",
                                             risk_score       = total_risk,
                                             clause_reference = clause.reference,
                                             suggested_fix    = self._get_risk_factor_fix(risk_factors_key, red_flag),
                                             contract_type    = self.contract_type.value,
                                             specific_text    = red_flag,
                                             legal_basis      = self._get_legal_basis(red_flag)
                                            )
                            )
        
        return terms
    

    def _analyze_cross_clause_issues(self, text: str, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Detect systemic issues spanning multiple clauses
        """
        terms = list()
        
        # Notice period imbalance (from your original but enhanced)
        notice_imbalance = self._check_notice_imbalance(clauses = clauses)
        if notice_imbalance:
            # Ensure the category used is a risk category
            notice_imbalance.category = self._map_to_risk_category(clause_category = "termination") 
            terms.append(notice_imbalance)
        
        # Missing reciprocal provisions
        missing_reciprocal = self._check_missing_reciprocal(text    = text, 
                                                            clauses = clauses,
                                                           )
        for item in missing_reciprocal:
            # Ensure the category used is a risk category
            item.category = self._map_to_risk_category(clause_category = "indemnification")
        terms.extend(missing_reciprocal)
        
        # Conflicting clauses
        conflicts = self._check_conflicting_clauses(clauses = clauses)
        for item in conflicts:
            # Ensure the category used is a risk category
            item.category = self._map_to_risk_category(clause_category = item.category) 
        terms.extend(conflicts)
        
        # One-sided discretionary powers
        one_sided_powers = self._check_one_sided_discretion(clauses = clauses)
        for item in one_sided_powers:
            # Ensure the category used is a risk category
            item.category = self._map_to_risk_category(clause_category = item.category)
        terms.extend(one_sided_powers)
        
        return terms
    

    def _analyze_missing_protections(self, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Analyze missing critical protections using PROTECTION_CHECKLIST
        """
        terms = list()
        
        for protection, config in self.risk_rules.PROTECTION_CHECKLIST.items():
            if not self._has_protection(clauses, protection, config['categories']):
                # For missing protections, map the first associated category to a risk category
                # This assumes config['categories'][0] is a clause category like "termination"
                risk_category = self._map_to_risk_category(clause_category = config['categories'][0]) if config['categories'] else "general"
                
                terms.append(UnfavorableTerm(term             = f"Missing Protection: {protection.replace('_', ' ').title()}",
                                             category         = risk_category,
                                             severity         = self._score_to_severity(config['risk_if_missing']),
                                             explanation      = f"Missing critical protection: {protection}. {self._get_missing_protection_explanation(protection)}",
                                             risk_score       = config['risk_if_missing'],
                                             suggested_fix    = self._get_missing_protection_fix(protection),
                                             contract_type    = self.contract_type.value,
                                             legal_basis      = f"Standard protection in {self.contract_type.value} contracts",
                                            )
                            )
        
        return terms
    

    def _analyze_against_benchmarks(self, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Compare terms against industry benchmarks
        """
        terms = list()
        
        for clause in clauses:
            benchmark_issues = self._check_benchmark_compliance(clause = clause)
            for item in benchmark_issues:
                # Ensure the category used is a risk category
                item.category = self._map_to_risk_category(clause_category = clause.category) 

            terms.extend(benchmark_issues)
        
        return terms
    

    def _check_notice_imbalance(self, clauses: List[ExtractedClause]) -> Optional[UnfavorableTerm]:
        """
        Enhanced notice period imbalance detection
        """
        term_clauses = [c for c in clauses if (c.category == "termination")]
        
        if not term_clauses:
            return None
        
        text             = " ".join([c.text for c in term_clauses])
        
        # Pattern matching for notice periods
        notice_patterns = [r'(\d+)\s*days?\s*notice',
                           r'notice\s*of\s*(\d+)\s*days',
                           r'(\d+)\s*days?\s*prior\s*notice',
                           r'written\s*notice\s*of\s*(\d+)\s*days',
                          ]
        
        all_periods     = list()

        for pattern in notice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_periods.extend([int(m) for m in matches])
        
        if (len(all_periods) >= 2):
            min_period = min(all_periods)
            max_period = max(all_periods)
            ratio      = max_period / min_period
            
            if (ratio >= 2):
                severity      = "critical" if (ratio >= 3) else "high"
                risk_score    = 80 if (ratio >= 3) else 60
                
                # Use the risk category mapping for termination
                risk_category = self._map_to_risk_category(clause_category = "termination")
                
                return UnfavorableTerm(term             = "Imbalanced Notice Periods",
                                       category         = risk_category,
                                       severity         = severity,
                                       explanation      = f"Significant notice period imbalance: {max_period} days vs {min_period} days (ratio: {ratio:.1f}x). Creates unfair burden.",
                                       risk_score       = risk_score,
                                       clause_reference = term_clauses[0].reference,
                                       suggested_fix    = f"Equalize notice periods to reasonable duration (e.g., 30 days mutual notice).",
                                       contract_type    = self.contract_type.value,
                                       benchmark_info   = f"Industry standard: Mutual 30-day notice periods",
                                      )
        
        return None
    

    def _check_missing_reciprocal(self, text: str, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Enhanced reciprocal provision analysis
        """
        terms         = list()
        
        # Check indemnification reciprocity
        indem_clauses = [c for c in clauses if (c.category == "indemnification")]
        
        if indem_clauses:
            has_one_sided = any(re.search(r'(you|employee|consultant|contractor)\s+shall\s+indemnify', c.text, re.IGNORECASE) for c in indem_clauses)
            has_mutual    = any("mutual" in c.text.lower() or "both parties" in c.text.lower() or "each party" in c.text.lower() for c in indem_clauses)
            
            if has_one_sided and not has_mutual:
                # Use the risk category mapping for indemnification
                risk_category = self._map_to_risk_category(clause_category = "indemnification")
                
                terms.append(UnfavorableTerm(term             = "Non-Reciprocal Indemnification",
                                             category         = risk_category,
                                             severity         = "critical",
                                             explanation      = "One-sided indemnification creates unlimited liability exposure without reciprocal protection.",
                                             risk_score       = 85,
                                             clause_reference = indem_clauses[0].reference,
                                             suggested_fix    = "Change to mutual indemnification: 'Each party shall indemnify the other for losses arising from their respective breach or negligence.'",
                                             contract_type    = self.contract_type.value,
                                             legal_basis      = "Mutuality of obligation principle",
                                            )
                            )
        
        return terms
    

    def _check_conflicting_clauses(self, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Detect conflicting clauses
        """
        terms       = list()
        
        # Group clauses by category for conflict analysis
        by_category = dict()

        for clause in clauses:
            # Map the clause category to the risk category for grouping purposes
            risk_cat = self._map_to_risk_category(clause_category = clause.category)
            if risk_cat not in by_category:
                by_category[risk_cat] = []
            
            by_category[risk_cat].append(clause)
        
        # Check for conflicts within each category
        for risk_category, category_clauses in by_category.items():
            if (len(category_clauses) >= 2):
                for i, clause1 in enumerate(category_clauses):
                    for clause2 in category_clauses[i+1:]:
                        if (self._are_clauses_conflicting(clause1, clause2)):
                            terms.append(UnfavorableTerm(term             = f"Conflicting {risk_category.title()} Clauses",
                                                         category         = risk_category,
                                                         severity         = "high",
                                                         explanation      = f"Clauses {clause1.reference} and {clause2.reference} contain conflicting terms creating legal ambiguity.",
                                                         risk_score       = 70,
                                                         clause_reference = f"{clause1.reference}, {clause2.reference}",
                                                         suggested_fix    = "Consolidate into single consistent clause or clarify precedence.",
                                                         contract_type    = self.contract_type.value,
                                                        )
                                        )
        
        return terms
    

    def _check_one_sided_discretion(self, clauses: List[ExtractedClause]) -> List[UnfavorableTerm]:
        """
        Check for one-sided discretionary powers
        """
        terms = list()
        
        for clause in clauses:
            text_lower = clause.text.lower()
            
            # Look for one-sided discretionary language
            if re.search(r'(sole|absolute|unfettered|unilateral)\s+(discretion|right|authority)', text_lower):
                if not re.search(r'(mutual|both parties|reasonable)\s+(discretion|agreement)', text_lower):
                    # Use the risk category mapping for the clause's category
                    risk_category = self._map_to_risk_category(clause_category = clause.category)
                    
                    terms.append(UnfavorableTerm(term             = "One-Sided Discretionary Power",
                                                 category         = risk_category,
                                                 severity         = "high",
                                                 explanation      = "Gives one party unilateral decision-making power without accountability standards.",
                                                 risk_score       = 75,
                                                 clause_reference = clause.reference,
                                                 suggested_fix    = "Change to 'reasonable discretion' or require 'mutual agreement'.",
                                                 contract_type    = self.contract_type.value,
                                                 legal_basis      = "Doctrine of good faith and fair dealing",
                                                )
                                )
        
        return terms
    

    def _check_benchmark_compliance(self, clause: ExtractedClause) -> List[UnfavorableTerm]:
        """
        Check clause against industry benchmarks
        """
        terms = list()
        
        # Non-compete duration benchmark
        if (clause.category == "non_compete"):
            duration_match = re.search(r'(\d+)\s*(month|year)', clause.text.lower())
            
            if duration_match:
                duration           = int(duration_match.group(1))
                unit               = duration_match.group(2)
                
                # Convert to months for comparison
                total_months       = duration * (12 if (unit == "year") else 1)
                
                benchmarks         = self.risk_rules.INDUSTRY_BENCHMARKS.get('non_compete_duration', {})
                industry_benchmark = benchmarks.get(self.contract_type.value, benchmarks.get('general', {}))
                
                if industry_benchmark:
                    reasonable = industry_benchmark.get('reasonable', 12)
                    excessive  = industry_benchmark.get('excessive', 24)
                    
                    if (total_months > excessive):
                        # Use the risk category mapping for non_compete
                        risk_category = self._map_to_risk_category(clause_category = clause.category)
                        
                        terms.append(UnfavorableTerm(term             = "Excessive Non-Compete Duration",
                                                     category         = risk_category,
                                                     severity         = "critical",
                                                     explanation      = f"{duration} {unit} non-compete exceeds industry excessive threshold of {excessive} months.",
                                                     risk_score       = 90,
                                                     clause_reference = clause.reference,
                                                     suggested_fix    = f"Reduce to {reasonable} months maximum.",
                                                     contract_type    = self.contract_type.value,
                                                     benchmark_info   = f"Industry standard: {reasonable} months reasonable, {excessive} months excessive",
                                                    )
                                    )
        
        return terms
    

    def _has_protection(self, clauses: List[ExtractedClause], protection: str, categories: List[str]) -> bool:
        """
        Check if protection exists in clauses
        """
        protection_patterns = {'for_cause_definition'     : ['for cause', 'cause defined', 'termination for cause', 'just cause'],
                               'severance_provision'      : ['severance', 'severance pay', 'termination benefits', 'separation pay'],
                               'mutual_indemnification'   : ['mutual indemnification', 'both parties indemnify', 'each party shall indemnify'],
                               'liability_cap'            : ['liability cap', 'limited liability', 'maximum liability', 'cap on damages'],
                               'prior_ip_exclusion'       : ['prior inventions', 'pre-existing ip', 'prior intellectual property', 'background ip'],
                               'confidentiality_duration' : ['confidentiality period', 'duration of confidentiality', 'term of confidentiality'],
                               'dispute_resolution'       : ['dispute resolution', 'arbitration', 'mediation', 'alternative dispute resolution'],
                               'change_control_process'   : ['change control', 'amendment process', 'modification procedure', 'change order'],
                               'insurance_requirements'   : ['insurance requirements', 'maintain insurance', 'proof of insurance'],
                               'force_majeure'            : ['force majeure', 'act of god', 'unforeseeable circumstances'],
                              }
        
        patterns            = protection_patterns.get(protection, [])
        relevant_clauses    = [c for c in clauses if not categories or c.category in categories]
        
        for clause in relevant_clauses:
            text_lower = clause.text.lower()
            if any(pattern in text_lower for pattern in patterns):
                return True
        
        return False
    

    # HELPER METHODS FOR EXPLANATIONS AND FIXES
    def _score_to_severity(self, score: float) -> str:
        """
        Convert risk score to severity level
        """
        if (score >= 80):
            return "critical"

        elif (score >= 60):
            return "high" 

        elif (score >= 40):
            return "medium"

        else:
            return "low"
    

    def _generate_pattern_explanation(self, pattern_desc: str, matched_text: str) -> str:
        """
        Generate explanation for pattern matches
        """
        explanations = {"Long duration restrictive covenant"     : f"Overly long restrictive period found: '{matched_text}'. May unreasonably restrict future employment.",
                        "Overly broad geographic/industry scope" : f"Excessively broad scope: '{matched_text}'. Could prevent working in entire industries or regions.",
                        "Unequal notice periods"                 : f"Imbalanced notice requirements: '{matched_text}'. Creates unfair advantage for one party.",
                        "Unlimited liability exposure"           : f"Uncapped liability: '{matched_text}'. Exposes to potentially catastrophic financial risk.",
                       }

        return explanations.get(pattern_desc, f"Risk pattern detected: {pattern_desc}")
    

    def _generate_pattern_fix(self, pattern_desc: str, category: str) -> str:
        """
        Generate fix suggestions for patterns
        """
        fixes = {"Long duration restrictive covenant"     : "Limit to 6-12 months maximum with reasonable geographic scope.",
                 "Overly broad geographic/industry scope" : "Narrow to specific competitors and reasonable geographic area.",
                 "Unequal notice periods"                 : "Equalize notice periods for both parties (e.g., 30 days mutual notice).",
                 "Unlimited liability exposure"           : "Add mutual liability cap (e.g., fees paid in preceding 12 months).",
                }

        return fixes.get(pattern_desc, "Review and modify to reasonable industry standards.")
    

    def _generate_keyword_explanation(self, keyword: str, category: str) -> str:
        """
        Generate explanations for keyword risks
        """
        explanations = {"non-compete"         : "Restrictive covenant limiting future employment opportunities.",
                        "unlimited liability" : "No cap on financial exposure - potentially catastrophic risk.",
                        "sole discretion"     : "Unilateral decision-making power without accountability.",
                        "at-will"             : "Termination without cause or protection - high job insecurity."
                       }

        return explanations.get(keyword, f"High-risk term '{keyword}' detected in {category} clause.")
    

    def _generate_keyword_fix(self, keyword: str, category: str) -> str:
        """
        Generate fixes for keyword risks
        """
        fixes = {"non-compete"         : "Limit duration to 12 months maximum and narrow geographic scope.",
                 "unlimited liability" : "Add mutual liability cap based on contract value.",
                 "sole discretion"     : "Change to 'reasonable discretion' or require 'mutual agreement'.",
                 "at-will"             : "Add 'for cause' definition and reasonable notice period.",
                }

        return fixes.get(keyword, "Modify to reasonable industry standards.")
    

    def _get_legal_basis(self, issue: str) -> str:
        """
        Get legal basis for risk issue
        """
        legal_bases = {"non-compete"         : "Reasonableness standard for restrictive covenants",
                       "unlimited liability" : "Unconscionability doctrine",
                       "sole discretion"     : "Doctrine of good faith and fair dealing", 
                       "at-will"             : "Employment protection statutes",
                       "unequal notice"      : "Mutuality of obligation principle",
                      }

        return legal_bases.get(issue, "General contract law principles")
    

    def _get_risk_factor_explanation(self, risk_category: str, red_flag: str) -> str:
        """
        Get explanation for risk factor red flags
        """
        explanations = {"restrictive_covenants": {"entire industry" : "Prohibits working in entire industry, not just direct competitors",
                                                  "worldwide"       : "Geographic scope is unreasonably broad",
                                                 }
                       }

        return explanations.get(risk_category, {}).get(red_flag, "Increases risk exposure")
    

    def _get_risk_factor_fix(self, risk_category: str, red_flag: str) -> str:
        """
        Get fix for risk factor issues
        """
        fixes = {"restrictive_covenants": {"entire industry" : "Limit to direct competitors only",
                                           "worldwide"       : "Narrow to specific geographic regions",
                                          }
                }

        return fixes.get(risk_category, {}).get(red_flag, "Modify to reasonable standards")
    

    def _get_missing_protection_explanation(self, protection: str) -> str:
        """
        Get explanation for missing protections
        """
        explanations = {"liability_cap"          : "No limit on potential financial damages",
                        "mutual_indemnification" : "One-sided liability protection",
                        "prior_ip_exclusion"     : "Could claim ownership of your existing work",
                        }

        return explanations.get(protection, "Critical protection missing from contract")
    

    def _get_missing_protection_fix(self, protection: str) -> str:
        """
        Get fix for missing protections
        """
        fixes = {"liability_cap"          : "Add mutual liability cap clause",
                 "mutual_indemnification" : "Add reciprocal indemnification",
                 "prior_ip_exclusion"     : "Add prior IP exclusion clause",
                }

        return fixes.get(protection, "Add appropriate protection clause")
    

    def _are_clauses_conflicting(self, clause1: ExtractedClause, clause2: ExtractedClause) -> bool:
        """
        Conflict detection between clauses
        """
        # Extract key numbers and terms
        nums1 = set(re.findall(r'\b\d+\b', clause1.text))
        nums2 = set(re.findall(r'\b\d+\b', clause2.text))
        
        # If both have numbers but no overlap, potential conflict
        if nums1 and nums2 and not nums1.intersection(nums2):
            return True
        
        # Check for contradictory language
        contradictions = [("shall", "shall not"),
                          ("must", "may not"), 
                          ("required", "prohibited"),
                         ]
        
        for positive, negative in contradictions:
            if (positive in clause1.text.lower() and negative in clause2.text.lower()) or (positive in clause2.text.lower() and negative in clause1.text.lower()):
                return True
        
        return False
    

    def _deduplicate_and_prioritize(self, terms: List[UnfavorableTerm]) -> List[UnfavorableTerm]:
        """
        Remove duplicates and sort by risk score
        """
        seen         = set()
        unique_terms = list()
        
        for term in terms:
            # Create unique key based on term, category, and specific text
            key = (term.term, term.category, term.specific_text)
            
            if key not in seen:
                seen.add(key)
                unique_terms.append(term)
        
        # Sort by risk score (descending)
        unique_terms.sort(key = lambda t: t.risk_score, reverse = True)
        
        # Return top 25 most critical terms
        return unique_terms[:25]
    

    def get_severity_distribution(self, terms: List[UnfavorableTerm]) -> Dict[str, int]:
        """
        Get distribution by severity
        """
        distribution = {"critical" : 0, 
                        "high"     : 0, 
                        "medium"   : 0, 
                        "low"      : 0,
                       }
        
        for term in terms:
            distribution[term.severity] = distribution.get(term.severity, 0) + 1
        
        log_info("Unfavorable terms severity distribution", **distribution)
        
        return distribution
    

    def get_category_distribution(self, terms: List[UnfavorableTerm]) -> Dict[str, int]:
        """
        Get distribution by category
        """
        categories   = [t.category for t in terms]
        distribution = dict(Counter(categories))
        
        log_info("Unfavorable terms category distribution", **distribution)
        
        return distribution
