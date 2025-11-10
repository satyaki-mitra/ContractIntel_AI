###########################
# test_legal_system.py
###########################

from legal_database.core_rules import CoreLegalRules
from legal_database.data_sources import LegalDataSources
from legal_database.jurisdictions import LegalDatabase, Jurisdiction

def test_complete_system():
    print("🧪 TESTING LEGAL DATABASE SYSTEM")
    print("=" * 50)
    
    # Test 1: Core Rules
    print("1. Testing Core Legal Rules...")
    core_rules = CoreLegalRules.get_core_rules()
    print(f"   ✅ Loaded {len(core_rules)} curated legal rules")
    
    # Test 2: Legal Database Integration
    print("\n2. Testing Legal Database Integration...")
    legal_db = LegalDatabase()
    print(f"   ✅ Database loaded with {len(legal_db.rules)} total rules")
    
    # Test 3: Enforceability Analysis
    print("\n3. Testing Enforceability Analysis...")
    sample_clause = {
        "text": "Employee agrees that company may withhold salary as penalty for breach",
        "category": "compensation",
        "risk_level": "high"
    }
    
    analysis = legal_db.assess_enforceability(sample_clause, Jurisdiction.INDIA)
    print(f"   ✅ Enforceability Score: {analysis['enforceability_score']}/100")
    print(f"   ✅ Likely Enforceable: {analysis['likely_enforceable']}")
    print(f"   ✅ Legal Issues: {analysis['legal_issues']}")
    
    # Test 4: External Sources
    print("\n4. Testing External Sources...")
    sources = LegalDataSources()
    precedents = sources.get_legal_precedent("india", "non_compete")
    print(f"   ✅ Found {len(precedents)} legal precedents")
    
    # Test 5: Database Update
    print("\n5. Testing Database Update...")
    success = sources.update_legal_database()
    print(f"   ✅ Database Update: {'Success' if success else 'Failed'}")
    
    print("\n" + "=" * 50)
    print("🎉 LEGAL DATABASE SYSTEM TEST COMPLETED!")

if __name__ == "__main__":
    test_complete_system()