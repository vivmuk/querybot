from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def setup_neo4j_indexes():
    """Set up Neo4j indexes and constraints."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            print("Setting up Neo4j indexes and constraints...")
            
            # Create unique constraint on Record.id
            try:
                session.run("CREATE CONSTRAINT record_id IF NOT EXISTS FOR (r:Record) REQUIRE r.id IS UNIQUE")
                print("✅ Created unique constraint on Record.id")
            except Exception as e:
                print(f"⚠️  Constraint creation failed (might already exist): {e}")
            
            # Create fulltext index for keyword search
            try:
                session.run("CALL db.index.fulltext.createNodeIndex('csv_fulltext', ['Record'], ['*'])")
                print("✅ Created fulltext index 'csv_fulltext'")
            except Exception as e:
                print(f"⚠️  Fulltext index creation failed (might already exist): {e}")
            
            # Check existing indexes
            result = session.run("SHOW INDEXES")
            indexes = [dict(record) for record in result]
            print(f"\n📋 Current indexes in database: {len(indexes)}")
            for idx in indexes:
                print(f"  - {idx.get('name', 'unnamed')}: {idx.get('type', 'unknown type')}")
            
            print("\n✅ Neo4j setup complete!")
            
    finally:
        driver.close()

if __name__ == "__main__":
    setup_neo4j_indexes() 