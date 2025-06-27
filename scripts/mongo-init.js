// MongoDB initialization script for AI Brain Python
// This script sets up the initial database structure and indexes

// Switch to the AI Brain database
db = db.getSiblingDB('ai_brain_dev');

// Create collections for cognitive systems
db.createCollection('cognitive_states');
db.createCollection('semantic_memory');
db.createCollection('goal_hierarchy');
db.createCollection('emotional_states');
db.createCollection('attention_states');
db.createCollection('confidence_tracking');
db.createCollection('cultural_contexts');
db.createCollection('skill_assessments');
db.createCollection('communication_protocols');
db.createCollection('temporal_plans');
db.createCollection('safety_assessments');
db.createCollection('monitoring_metrics');
db.createCollection('tool_validations');
db.createCollection('workflow_states');
db.createCollection('multimodal_data');
db.createCollection('human_feedback');

// Create indexes for cognitive_states collection
db.cognitive_states.createIndex({ "system_id": 1 });
db.cognitive_states.createIndex({ "timestamp": 1 });
db.cognitive_states.createIndex({ "user_id": 1 });
db.cognitive_states.createIndex({ "system_id": 1, "timestamp": -1 });

// Create vector search index for semantic_memory
db.semantic_memory.createIndex({ 
    "embedding": "2dsphere" 
});
db.semantic_memory.createIndex({ "content": "text" });
db.semantic_memory.createIndex({ "timestamp": 1 });
db.semantic_memory.createIndex({ "relevance_score": -1 });
db.semantic_memory.createIndex({ "access_count": -1 });

// Create indexes for goal_hierarchy
db.goal_hierarchy.createIndex({ "user_id": 1 });
db.goal_hierarchy.createIndex({ "status": 1 });
db.goal_hierarchy.createIndex({ "priority": -1 });
db.goal_hierarchy.createIndex({ "created_at": 1 });
db.goal_hierarchy.createIndex({ "updated_at": 1 });

// Create indexes for emotional_states
db.emotional_states.createIndex({ "user_id": 1 });
db.emotional_states.createIndex({ "primary_emotion": 1 });
db.emotional_states.createIndex({ "timestamp": 1 });
db.emotional_states.createIndex({ "confidence": -1 });

// Create indexes for attention_states
db.attention_states.createIndex({ "user_id": 1 });
db.attention_states.createIndex({ "focus_level": -1 });
db.attention_states.createIndex({ "timestamp": 1 });

// Create indexes for monitoring_metrics
db.monitoring_metrics.createIndex({ "metric_type": 1 });
db.monitoring_metrics.createIndex({ "timestamp": 1 });
db.monitoring_metrics.createIndex({ "system_id": 1 });
db.monitoring_metrics.createIndex({ "value": -1 });

// Create indexes for safety_assessments
db.safety_assessments.createIndex({ "assessment_type": 1 });
db.safety_assessments.createIndex({ "risk_level": -1 });
db.safety_assessments.createIndex({ "timestamp": 1 });
db.safety_assessments.createIndex({ "is_safe": 1 });

// Create compound indexes for performance
db.cognitive_states.createIndex({ 
    "user_id": 1, 
    "system_id": 1, 
    "timestamp": -1 
});

db.semantic_memory.createIndex({ 
    "user_id": 1, 
    "relevance_score": -1, 
    "timestamp": -1 
});

// Create TTL index for temporary data (expires after 30 days)
db.monitoring_metrics.createIndex(
    { "timestamp": 1 }, 
    { expireAfterSeconds: 2592000 }  // 30 days
);

// Create user for AI Brain application
db.createUser({
    user: "ai_brain_user",
    pwd: "ai_brain_password",
    roles: [
        {
            role: "readWrite",
            db: "ai_brain_dev"
        }
    ]
});

// Insert sample data for testing
db.cognitive_states.insertOne({
    system_id: "emotional_intelligence",
    user_id: "test_user",
    state: {
        primary_emotion: "neutral",
        confidence: 0.8,
        intensity: 0.5
    },
    timestamp: new Date(),
    metadata: {
        version: "1.0.0",
        source: "initialization"
    }
});

db.semantic_memory.insertOne({
    content: "This is a test memory for the AI Brain system initialization.",
    embedding: [0.1, 0.2, 0.3, 0.4, 0.5],  // Sample embedding vector
    metadata: {
        source: "initialization",
        type: "test_memory"
    },
    relevance_score: 1.0,
    access_count: 0,
    timestamp: new Date()
});

print("AI Brain MongoDB initialization completed successfully!");
print("Created collections and indexes for all cognitive systems.");
print("Created application user: ai_brain_user");
print("Inserted sample test data.");

// Display collection statistics
print("\nCollection Statistics:");
db.getCollectionNames().forEach(function(collection) {
    var count = db.getCollection(collection).countDocuments();
    print(collection + ": " + count + " documents");
});
