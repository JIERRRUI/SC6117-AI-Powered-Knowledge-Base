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
// Use the correct Gemini embedding model name
const EMBEDDING_MODEL = "text-embedding-004";

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

// Helper to split array into chunks for concurrency
const chunkArray = <T>(array: T[], size: number): T[][] => {
  const chunked: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunked.push(array.slice(i, i + size));
  }
  return chunked;
};

/**
 * Generate embeddings for a batch of notes using Gemini API
 * Updated for Concurrency: Processes 10 notes at a time
 * @param notes Notes to embed
 * @returns Array of note embeddings
 */
export const generateEmbeddingsBatch = async (
  notes: Note[]
): Promise<NoteEmbedding[]> => {
  const ai = getAIClient();
  const embeddings: NoteEmbedding[] = [];

  // Batch size of 10 for reasonable concurrency without rate limiting aggressively
  const chunks = chunkArray(notes, 10);

  console.log(
    `Starting embedding generation for ${notes.length} notes in ${chunks.length} batches...`
  );

  for (const [index, chunk] of chunks.entries()) {
    console.log(
      `Processing batch ${index + 1}/${chunks.length} (${
        chunk.length
      } notes)...`
    );

    // Create array of promises for this chunk
    const chunkPromises = chunk.map(async (note) => {
      try {
        const text = `${note.title}\n${note.content.substring(0, 500)}`;

        // Use the correct API format for @google/genai
        const response = await ai.models.embedContent({
          model: EMBEDDING_MODEL,
          contents: text,
        });

        // Handle both possible response shapes
        const embedding =
          (response as any).embedding ?? (response as any).embeddings?.[0];

        if (embedding?.values?.length) {
          return {
            noteId: note.id,
            vector: embedding.values,
            timestamp: Date.now(),
            modelUsed: EMBEDDING_MODEL,
            textLength: text.length,
          } as NoteEmbedding;
        } else {
          console.error(`❌ No embedding values for note ${note.id}`);
          return null;
        }
      } catch (err) {
        console.error(`❌ Failed to embed note ${note.id}:`, err);
        return null;
      }
    });

    // Wait for all requests in this chunk to finish
    const results = await Promise.all(chunkPromises);

    // Filter out nulls and add to main list
    const validResults = results.filter((r): r is NoteEmbedding => r !== null);
    embeddings.push(...validResults);

    // Optional: Small delay between chunks to be nice to the API
    if (index < chunks.length - 1) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }

  console.log(
    `Generated embeddings for ${embeddings.length}/${notes.length} notes`
  );
  return embeddings;
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
  const assignedNotes = new Set<string>(); // Notes assigned to a valid cluster

  // Find similar pairs
  const pairs = findSimilarPairs(embeddings, similarityThreshold);

  console.log(
    `Found ${pairs.length} similar pairs at threshold ${similarityThreshold}`
  );
  if (pairs.length > 0) {
    console.log(
      `Top 5 similarities:`,
      pairs
        .slice(0, 5)
        .map(
          (p) =>
            `${p.note1Id.slice(0, 8)}-${p.note2Id.slice(
              0,
              8
            )}: ${p.similarity.toFixed(3)}`
        )
    );
  }

  // Build adjacency graph
  const graph = new Map<string, Set<string>>();
  embeddings.forEach((e) => {
    graph.set(e.noteId, new Set());
  });

  pairs.forEach((pair) => {
    graph.get(pair.note1Id)?.add(pair.note2Id);
    graph.get(pair.note2Id)?.add(pair.note1Id);
  });

  // Cluster using connected components (BFS)
  const findConnectedComponent = (startId: string): Set<string> => {
    const component = new Set<string>();
    const queue = [startId];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (component.has(current)) continue;

      component.add(current);

      const neighbors = graph.get(current) || new Set();
      neighbors.forEach((neighbor) => {
        if (!component.has(neighbor)) {
          queue.push(neighbor);
        }
      });
    }

    return component;
  };

  // Find all connected components (clusters)
  const visited = new Set<string>();
  embeddings.forEach((e) => {
    if (!visited.has(e.noteId)) {
      const component = findConnectedComponent(e.noteId);

      // Mark all notes in this component as visited
      component.forEach((id) => visited.add(id));

      // Only create partition if it meets minimum size
      if (component.size >= minClusterSize) {
        const clusterEmbeddings = Array.from(component)
          .map((id) => embeddings.find((e) => e.noteId === id)!)
          .filter((e) => e) // safety check
          .map((e) => e.vector);

        partitions.push({
          id: `partition-${partitions.length}`,
          noteIds: Array.from(component),
          centroid: computeCentroid(clusterEmbeddings),
          silhouetteScore: 0,
        });

        // Mark as assigned
        component.forEach((id) => assignedNotes.add(id));
      }
    }
  });

  // IMPORTANT: Handle unassigned notes (singletons and small groups)
  // These notes didn't meet minClusterSize but should still be clustered
  const unassigned = embeddings.filter((e) => !assignedNotes.has(e.noteId));

  if (unassigned.length > 0) {
    console.log(
      `⚠️ ${unassigned.length} notes not assigned to clusters, will be grouped in Phase 3`
    );

    // DON'T create individual singleton partitions - they'll be handled in clusteringService
    // where unclustered notes get added to an "Additional Topics" cluster
    // This is better than having many 1-note clusters
  }

  console.log(`Embedding-guided partitions created:`);
  partitions.forEach((p, idx) => {
    console.log(`  Partition ${idx}: ${p.noteIds.length} notes`);
  });
  console.log(
    `Total: ${partitions.length} partitions covering ${assignedNotes.size}/${embeddings.length} notes`
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
