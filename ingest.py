# ingest.py
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

QDRANT_HOST = "http://localhost:6333"

# 1) load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

# 2) initialize qdrant client
client = QdrantClient(url=QDRANT_HOST)

COLLECTION_NAME = "my_docs"
VECTOR_SIZE = embedder.get_sentence_embedding_dimension()

# create collection if missing
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=VECTOR_SIZE, distance=rest.Distance.COSINE  # type: ignore
        ),
    )


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_documents(doc_texts):
    points = []
    for doc_id, text in enumerate(tqdm(doc_texts, desc="docs")):
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            pid = str(uuid.uuid4())
            payload = {
                "doc_id": doc_id,
                "chunk_idx": idx,
                "text": chunk[:1000],  # store snippet
            }
            points.append(
                rest.PointStruct(id=pid, vector=emb.tolist(), payload=payload)
            )
            # break into batches if many points
    # upsert in batches
    BATCH = 128
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + BATCH])


if __name__ == "__main__":
    # Example: replace with actual loader — read files, PDFs, etc.
    docs = [
        """
    Feature flags, also known as feature toggles, allow teams to enable or disable specific
    functionality in a production environment without deploying new code. They decouple deployment
    from release by providing runtime control. Teams use them for gradual rollouts, A/B testing,
    and controlled exposure of new features to small segments of users. If an unexpected issue
    occurs, a feature can be rolled back instantly by flipping the flag rather than redeploying.
    Feature flags support different strategies such as boolean flags, multivariate flags, and
    permission-based user targeting.
    """,
        """
    In a microservices architecture, applications are decomposed into independently deployable
    services that communicate over lightweight protocols such as HTTP or gRPC. Each service owns
    its own data and represents a specific business capability. This improves scalability and
    deployment flexibility but introduces operational challenges such as distributed tracing,
    network latency, and service discovery. Teams often use API gateways, service meshes, and
    centralized logging to manage cross-cutting concerns. Microservices can be deployed and scaled
    individually, enabling teams to iterate faster with reduced blast radius.
    """,
        """
    Rate limiting is a technique used to control how many requests a client or IP address can make
    to a server within a defined timeframe. This prevents abuse, ensures fair usage, and protects
    backend systems from overload. Common strategies include token bucket, leaky bucket, fixed
    window, sliding window, and concurrency limits. Rate limiting rules can be applied globally,
    per user, per API key, or per route. Many distributed systems rely on Redis, Memcached, or
    specialized proxies like Envoy or NGINX to maintain counters and enforce limits efficiently. 
    Proper rate limiting helps prevent denial-of-service (DoS) scenarios.
    """,
        """
    Caching improves application performance by storing frequently accessed data in fast storage
    layers such as Redis, Memcached, CDN edges, or in-memory caches. Caches reduce database load,
    improve response times, and lower infrastructure costs. Cache strategies include write-through,
    write-back, write-around, and cache invalidation policies like TTL-based eviction or LRU.
    Developers must handle cache consistency, cache stampedes, and warm-up strategies. Effective
    caching can reduce response latency from hundreds of milliseconds to single-digit milliseconds.
    """,
        """
    Apache Kafka is a distributed event streaming platform used for high-throughput messaging,
    real-time data pipelines, and log aggregation. Kafka stores events in partitions within topics,
    and consumers read data sequentially using offsets. Kafka brokers replicate partitions across
    nodes to ensure fault tolerance. Producers write events to Kafka asynchronously, enabling loose
    coupling between systems. Common patterns include event sourcing, CQRS, streaming ETL, and
    change-data-capture (CDC). Kafka handles millions of messages per second and powers modern data
    architectures for analytics, monitoring, and event-driven microservices.
    """,
        """
    My name is Himanth Godari, and I am a backend developer with strong experience in
    building scalable, reliable backend systems. I actively work with technologies such
    as Node.js, TypeScript, Rust, PostgreSQL, Docker, and gRPC. I focus on designing
    clean architectures, resilient system components, and distributed service patterns.
    Recently, I have been diving deeper into AI engineering, especially practical
    Retrieval-Augmented Generation (RAG), vector databases, embeddings, and local LLM
    inference. My goal is to combine strong backend fundamentals with modern AI system
    design to build intelligent, production-grade systems.
    """,
        """
    One of my major hands-on projects involved building a custom HTTP server in Rust
    from scratch. This included request parsing, routing, and connection handling.
    I implemented a custom thread pool, graceful shutdown logic, static file serving,
    and automated tests. This project helped me understand low-level network behavior,
    memory management, and concurrent execution models.
    """,
        """
    I built a fully functional Retrieval-Augmented Generation system using Qdrant as the
    vector database, SentenceTransformers for embeddings, and local models like Phi-3
    Mini and TinyLLaMA for inference. The system performs document chunking, semantic
    embeddings, vector similarity search, reranking, and prompt assembly. Running LLMs
    locally on limited GPU hardware helped me understand quantization strategies,
    performance tradeoffs, prompt engineering, and grounding responses to context.
    """,
        """
    I developed a distributed microservices architecture using Node.js and gRPC. This
    involved designing protobuf schemas, handling unary and streaming RPCs, integrating
    load balancing, and observing request flows through logs and metrics. The system
    demonstrated how microservices can be decoupled, scaled independently, and managed
    effectively with proper communication protocols and schema evolution planning.
    """,
        """
    I implemented a secure OAuth2 and JWT authentication server using Node.js. The
    project includes access tokens, refresh tokens, session revocation, and audit
    trails. It follows secure token rotation practices and adds middleware for rate
    limiting, role-based access, and error handling. This system is designed to be
    embedded into larger microservice architectures.
    """,
        """
    I worked on a real-world Leegality e-sign integration using webhook-based event
    processing. The system verifies signatures, validates payload integrity, stores
    signed documents, and triggers business workflows automatically based on document
    lifecycle updates. This project strengthened my understanding of asynchronous
    orchestration and secure webhook consumption.
    """,
        """
    I built an encryption pipeline using Node.js crypto and crypto-js that supports a
    hybrid RSA + AES workflow. Each client receives a unique public key. The payload is
    encrypted using AES for performance, while the AES key is protected using RSA. The
    goal was secure, efficient, end-to-end payload transmission between distributed
    systems. This project helped me understand key exchange patterns and cryptographic
    best practices.
    """,
        """
    I created a database indexing and analytics project to learn how indexing strategies
    affect performance. The project involved creating schemas for a blog analytics
    system, adding B-tree, GIN, hash, and partial indexes, and benchmarking query
    performance. I used EXPLAIN ANALYZE to study query planners, index scans, and
    performance bottlenecks under large datasets.
    """,
        """
    I implemented a multi-service distributed system using gRPC to explore
    service-to-service communication, schema evolution, and load distribution. Each
    service represented a domain boundary and communicated through strongly typed
    protobuf definitions. The project deepened my understanding of distributed tracing,
    request streaming, concurrency limits, and error propagation in microservice
    ecosystems.
    """,
        """
    My current learning goals include advanced RAG architectures, efficient embeddings,
    vector database optimizations, quantization-aware LLM inference, and containerized
    AI pipelines using Docker. I am focused on building backend systems that integrate
    AI as a native capability — providing context-aware, intelligent responses across
    applications. I aim to grow into a backend + AI hybrid engineer with deep knowledge
    of distributed systems and language models.
    """,
    ]
    ingest_documents(docs)
    print("Ingest complete.")
