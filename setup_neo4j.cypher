// Create unique constraint on Record.id
CREATE CONSTRAINT record_id IF NOT EXISTS ON (r:Record) ASSERT r.id IS UNIQUE;

// Create fulltext index for keyword search
CALL db.index.fulltext.createNodeIndex('csv_fulltext', ['Record'], ['*']);

// Note: Vector index commented out since embeddings are not available
// CALL db.index.vector.createGraphNodeIndex('record_embeddings', 'Record', 'vec', 384, 'cosine'); 