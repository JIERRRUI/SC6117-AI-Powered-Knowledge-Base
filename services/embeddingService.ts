import {
  Note,
  NoteEmbedding,
  EmbeddingIndex,
  SimilarityPair,
  EmbeddingPartition,
} from "../types";
import { GoogleGenAI } from "@google/genai";

const getAIClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// --- Embedding Cache Management ---

const EMBEDDING_CACHE_KEY = "embedding_index_v1";
const EMBEDDING_MODEL = "text-embedding-004"; // Gemini's embedding model

export const loadEmbeddingIndex = (): EmbeddingIndex => {
  try {
    const raw = localStorage.getItem(EMBEDDING_CACHE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as EmbeddingIndex;
      if (parsed.embeddings && Array.isArray(parsed.embeddings)) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn("Failed to load embedding index:", e);
  }

  return {
    embeddings: [],
    lastUpdated: 0,
    modelVersion: EMBEDDING_MODEL,
    totalNotes: 0,
  };
};

const saveEmbeddingIndex = (index: EmbeddingIndex): void => {
  try {
    localStorage.setItem(EMBEDDING_CACHE_KEY, JSON.stringify(index));
  } catch (e) {
    console.warn("Failed to save embedding index:", e);
  }
};

// --- Embedding Generation (Step 2.1) ---

/**
 * Generate embeddings for a batch of notes using Gemini API
 * @param notes Notes to embed
 * @returns Array of note embeddings
 */
export const generateEmbeddingsBatch = async (
  notes: Note[]
): Promise<NoteEmbedding[]> => {
  const ai = getAIClient();
  const embeddings: NoteEmbedding[] = [];

  // Prepare texts: combine title + snippet of content
  const texts = notes.map((n) => {
    const snippet = n.content.substring(0, 500); // First 500 chars
    return `${n.title}\n${snippet}`;
  });

  try {
    // Call Gemini embedding API
    const response = await ai.models.embedContent({
      model: EMBEDDING_MODEL,
      content: {
        parts: texts.map((text) => ({ text })),
      },
    });

    // Process embeddings
    const embeddingsList = (response as any).embeddings || [];

    notes.forEach((note, index) => {
      if (embeddingsList[index]) {
        embeddings.push({
          noteId: note.id,
          vector: embeddingsList[index].values || [],
          timestamp: Date.now(),
          modelUsed: EMBEDDING_MODEL,
          textLength: texts[index].length,
        });
      }
    });

    console.log(
      `Generated embeddings for ${embeddings.length}/${notes.length} notes`
    );
    return embeddings;
  } catch (e) {
    console.error("Failed to generate embeddings:", e);
    return [];
  }
};

/**
 * Get or generate embeddings for notes with caching
 * @param notes Notes to embed
 * @param forceRefresh Skip cache and regenerate
 * @returns Embeddings for each note
 */
export const getOrGenerateEmbeddings = async (
  notes: Note[],
  forceRefresh: boolean = false
): Promise<NoteEmbedding[]> => {
  const index = loadEmbeddingIndex();
  const now = Date.now();
  const CACHE_TTL = 7 * 24 * 60 * 60 * 1000; // 7 days

  // Check if cache is still valid
  if (
    !forceRefresh &&
    index.embeddings.length > 0 &&
    now - index.lastUpdated < CACHE_TTL
  ) {
    // Return cached embeddings for available notes
    const cached = index.embeddings.filter((e) =>
      notes.some((n) => n.id === e.noteId)
    );

    // Find missing notes
    const cachedIds = new Set(cached.map((e) => e.noteId));
    const missing = notes.filter((n) => !cachedIds.has(n.id));

    if (missing.length === 0) {
      console.log(`Cache hit: ${cached.length} embeddings from cache`);
      return cached;
    }

    // Generate embeddings for missing notes
    console.log(
      `Partial cache hit: ${cached.length} cached, generating ${missing.length} new`
    );
    const newEmbeddings = await generateEmbeddingsBatch(missing);

    // Update index
    const updated = [...cached, ...newEmbeddings];
    const newIndex: EmbeddingIndex = {
      embeddings: updated,
      lastUpdated: now,
      modelVersion: EMBEDDING_MODEL,
      totalNotes: updated.length,
    };
    saveEmbeddingIndex(newIndex);

    return updated;
  }

  // Generate all embeddings
  console.log(
    `Cache miss or refresh forced: generating ${notes.length} embeddings`
  );
  const newEmbeddings = await generateEmbeddingsBatch(notes);

  // Save to cache
  const newIndex: EmbeddingIndex = {
    embeddings: newEmbeddings,
    lastUpdated: now,
    modelVersion: EMBEDDING_MODEL,
    totalNotes: newEmbeddings.length,
  };
  saveEmbeddingIndex(newIndex);

  return newEmbeddings;
};

// --- Similarity Computation ---

/**
 * Compute cosine similarity between two vectors
 */
const cosineSimilarity = (vec1: number[], vec2: number[]): number => {
  if (vec1.length === 0 || vec2.length === 0) return 0;

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (norm1 * norm2);
};

/**
 * Compute centroid (average) of embedding vectors
 */
const computeCentroid = (vectors: number[][]): number[] => {
  if (vectors.length === 0) return [];

  const dim = vectors[0].length;
  const centroid = new Array(dim).fill(0);

  vectors.forEach((vec) => {
    for (let i = 0; i < dim; i++) {
      centroid[i] += vec[i];
    }
  });

  for (let i = 0; i < dim; i++) {
    centroid[i] /= vectors.length;
  }

  return centroid;
};

/**
 * Find all high-similarity pairs using threshold
 * @param embeddings Embeddings to compare
 * @param threshold Similarity threshold (0-1)
 * @returns Pairs of similar notes
 */
export const findSimilarPairs = (
  embeddings: NoteEmbedding[],
  threshold: number = 0.7
): SimilarityPair[] => {
  const pairs: SimilarityPair[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const similarity = cosineSimilarity(
        embeddings[i].vector,
        embeddings[j].vector
      );

      if (similarity >= threshold) {
        pairs.push({
          note1Id: embeddings[i].noteId,
          note2Id: embeddings[j].noteId,
          similarity,
        });
      }
    }
  }

  return pairs.sort((a, b) => b.similarity - a.similarity);
};

// --- Embedding-Guided Partitioning (Step 2.2) ---

/**
 * Simple clustering using cosine similarity (DBSCAN-like approach)
 * @param embeddings Note embeddings
 * @param similarityThreshold Threshold for grouping (0-1)
 * @param minClusterSize Minimum notes per cluster
 * @returns Partitions (clusters) based on embeddings
 */
export const embeddingGuidedPartitioning = (
  embeddings: NoteEmbedding[],
  similarityThreshold: number = 0.65,
  minClusterSize: number = 2
): EmbeddingPartition[] => {
  const partitions: EmbeddingPartition[] = [];
  const visited = new Set<string>();

  // Find similar pairs
  const pairs = findSimilarPairs(embeddings, similarityThreshold);

  // Build adjacency
  const graph = new Map<string, Set<string>>();
  embeddings.forEach((e) => {
    if (!graph.has(e.noteId)) {
      graph.set(e.noteId, new Set());
    }
  });

  pairs.forEach((pair) => {
    graph.get(pair.note1Id)?.add(pair.note2Id);
    graph.get(pair.note2Id)?.add(pair.note1Id);
  });

  // Cluster using connected components
  const clusterNotes = (startId: string, cluster: Set<string>): void => {
    const queue = [startId];

    while (queue.length > 0) {
      const current = queue.shift()!;

      if (visited.has(current) || cluster.has(current)) continue;

      cluster.add(current);
      visited.add(current);

      const neighbors = graph.get(current) || new Set();
      neighbors.forEach((neighbor) => {
        if (!visited.has(neighbor)) {
          queue.push(neighbor);
        }
      });
    }
  };

  // Find all clusters
  embeddings.forEach((e) => {
    if (!visited.has(e.noteId)) {
      const cluster = new Set<string>();
      clusterNotes(e.noteId, cluster);

      // Only add if meets minimum size
      if (cluster.size >= minClusterSize) {
        const clusterEmbeddings = Array.from(cluster)
          .map((id) => embeddings.find((e) => e.noteId === id)!)
          .map((e) => e.vector);

        partitions.push({
          id: `partition-${partitions.length}`,
          noteIds: Array.from(cluster),
          centroid: computeCentroid(clusterEmbeddings),
          silhouetteScore: 0, // TODO: compute actual silhouette score
        });
      }
    }
  });

  // Singletons become their own partition
  const unvisited = embeddings.filter((e) => !visited.has(e.noteId));
  unvisited.forEach((e) => {
    partitions.push({
      id: `partition-${partitions.length}`,
      noteIds: [e.noteId],
      centroid: e.vector,
      silhouetteScore: 0,
    });
  });

  console.log(
    `Created ${partitions.length} partitions from ${embeddings.length} notes`
  );
  return partitions;
};

/**
 * Get embedding statistics for analysis
 */
export const getEmbeddingStats = (
  embeddings: NoteEmbedding[]
): Record<string, any> => {
  if (embeddings.length === 0) {
    return { totalEmbeddings: 0, avgVectorDim: 0 };
  }

  const vectorDims = embeddings.map((e) => e.vector.length);
  const avgDim = vectorDims.reduce((a, b) => a + b, 0) / vectorDims.length;

  return {
    totalEmbeddings: embeddings.length,
    avgVectorDim: Math.round(avgDim),
    avgTextLength: Math.round(
      embeddings.reduce((sum, e) => sum + e.textLength, 0) / embeddings.length
    ),
    modelUsed: embeddings[0]?.modelUsed || "unknown",
    cacheAge: Math.round((Date.now() - embeddings[0]?.timestamp) / 1000 / 60), // minutes
  };
};
