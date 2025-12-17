import {
  Note,
  ClusterNode,
  ClusteringDecision,
  ClusteringMemory,
  MemoryBuffer,
  EmbeddingPartition,
  SemanticCentroid,
  HardSample,
  SupervisionSignal,
  ConstraintSet,
  SemanticClusteringResult,
} from "../types";
import {
  clusterNotesWithGemini,
  dualPromptClusterNotesWithGemini,
  generateSemanticCentroid,
  detectHardSample,
} from "./geminiService";
import {
  getOrGenerateEmbeddings,
  embeddingGuidedPartitioning,
  getEmbeddingStats,
} from "./embeddingService";

// Minimal, incremental clustering scaffolding with caching.
// Vector-store integration and embeddings are planned but not required to run.

export interface IngestionResult {
  notes: Note[];
  changedNoteIds: string[];
}

// Simple content hash to detect changes (not cryptographic).
const fastHash = (s: string) => {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h.toString(16);
};

const CACHE_KEY = "clusters_cache_v1";
const HASH_INDEX_KEY = "note_hash_index_v1";
const MEMORY_KEY = "clustering_memory_v1";
const MAX_MEMORY_SIZE = 50; // Keep last 50 clustering decisions

// --- Dynamic Memory System Implementation ---

class ClusteringMemoryBuffer implements MemoryBuffer {
  private memory: ClusteringMemory;

  constructor() {
    this.memory = this.loadMemory();
  }

  private loadMemory(): ClusteringMemory {
    try {
      const raw = localStorage.getItem(MEMORY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as ClusteringMemory;
        // Validate structure
        if (parsed.decisions && Array.isArray(parsed.decisions)) {
          return parsed;
        }
      }
    } catch (e) {
      console.warn("Failed to load clustering memory:", e);
    }

    // Return default empty memory
    return {
      decisions: [],
      lastClusteringTime: 0,
      totalNotesProcessed: 0,
      averageConfidence: 0,
      version: "1.0",
    };
  }

  private saveMemory(): void {
    try {
      localStorage.setItem(MEMORY_KEY, JSON.stringify(this.memory));
    } catch (e) {
      console.warn("Failed to save clustering memory:", e);
    }
  }

  addDecision(decision: ClusteringDecision): void {
    this.memory.decisions.push(decision);

    // Maintain size limit
    if (this.memory.decisions.length > MAX_MEMORY_SIZE) {
      this.memory.decisions = this.memory.decisions.slice(-MAX_MEMORY_SIZE);
    }

    // Update stats
    this.memory.lastClusteringTime = decision.timestamp;
    this.memory.totalNotesProcessed += decision.noteIds.length;

    // Recalculate average confidence
    const confidences = this.memory.decisions.map((d) => d.confidence);
    this.memory.averageConfidence =
      confidences.reduce((a, b) => a + b, 0) / confidences.length;

    this.saveMemory();
  }

  getRecentDecisions(limit: number = 10): ClusteringDecision[] {
    return this.memory.decisions.slice(-limit);
  }

  getDecisionById(id: string): ClusteringDecision | null {
    return this.memory.decisions.find((d) => d.id === id) || null;
  }

  getMemoryForNotes(noteIds: string[]): ClusteringMemory {
    const relevantDecisions = this.memory.decisions.filter((decision) =>
      decision.noteIds.some((id) => noteIds.includes(id))
    );

    return {
      ...this.memory,
      decisions: relevantDecisions,
    };
  }

  clearOldDecisions(keepLastN: number = 10): void {
    if (this.memory.decisions.length > keepLastN) {
      this.memory.decisions = this.memory.decisions.slice(-keepLastN);
      this.saveMemory();
    }
  }

  getStats() {
    return {
      totalDecisions: this.memory.decisions.length,
      avgConfidence: this.memory.averageConfidence,
      lastUpdate: this.memory.lastClusteringTime,
    };
  }
}

// Global memory buffer instance
const memoryBuffer = new ClusteringMemoryBuffer();

// Export memory buffer for use in other services
export { memoryBuffer };

export const readCachedClusters = (): ClusterNode[] => {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    return raw ? (JSON.parse(raw) as ClusterNode[]) : [];
  } catch {
    return [];
  }
};

export const writeCachedClusters = (clusters: ClusterNode[]): void => {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(clusters));
  } catch {
    // ignore
  }
};

export const readHashIndex = (): Record<string, string> => {
  try {
    const raw = localStorage.getItem(HASH_INDEX_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
};

export const writeHashIndex = (idx: Record<string, string>): void => {
  try {
    localStorage.setItem(HASH_INDEX_KEY, JSON.stringify(idx));
  } catch {
    // ignore
  }
};

// Ingest notes and detect which changed since last run.
export const ingestNotes = (notes: Note[]): IngestionResult => {
  const idx = readHashIndex();
  const changed: string[] = [];

  for (const n of notes) {
    const h = fastHash(`${n.title}\n${n.content}`);
    if (idx[n.id] !== h) {
      idx[n.id] = h;
      changed.push(n.id);
    }
  }

  writeHashIndex(idx);
  return { notes, changedNoteIds: changed };
};

// Compute a hash of cluster assignments to detect convergence
const hashClusterAssignments = (clusters: ClusterNode[]): string => {
  const assignments = clusters
    .map((c) => c.id + ":" + (c.children?.map((ch) => ch.id).join(",") || ""))
    .join("|");
  let h = 0;
  for (let i = 0; i < assignments.length; i++)
    h = (h * 31 + assignments.charCodeAt(i)) >>> 0;
  return h.toString(16);
};

// Iterative clustering with convergence detection (Step 1.3)
export const iterativeClusterWithRefinement = async (
  notes: Note[],
  maxIterations: number = 3
): Promise<{
  clusters: ClusterNode[];
  iterations: number;
  converged: boolean;
}> => {
  let clusters: ClusterNode[] = [];
  let previousHash = "";
  let iterations = 0;

  for (let i = 0; i < maxIterations; i++) {
    iterations = i + 1;

    // Perform dual-prompt clustering with memory feedback
    const result = await dualPromptClusterNotesWithGemini(notes, memoryBuffer);
    clusters = result.clusters;

    // Check for convergence
    const currentHash = hashClusterAssignments(clusters);
    if (currentHash === previousHash) {
      // Converged
      return { clusters, iterations, converged: true };
    }

    previousHash = currentHash;
  }

  // Max iterations reached
  return { clusters, iterations, converged: false };
};

// Incremental clustering: if few notes changed, you can re-cluster selectively.
// For now, we call the existing Gemini clustering across all notes, but
// preserve caching and return immediately if nothing changed.
export const incrementalCluster = async (
  notes: Note[]
): Promise<ClusterNode[]> => {
  const { changedNoteIds } = ingestNotes(notes);
  if (changedNoteIds.length === 0) {
    const cached = readCachedClusters();
    if (cached.length > 0) return cached;
  }

  // Fallback: full clustering via Gemini service (existing logic).
  const clusters = await clusterNotesWithGemini(notes);
  writeCachedClusters(clusters);
  return clusters;
};

// Enhanced clustering with dual-prompt and iterative refinement (Phase 1)
export const enhancedCluster = async (
  notes: Note[],
  useIterativeRefinement: boolean = true
): Promise<ClusterNode[]> => {
  const { changedNoteIds } = ingestNotes(notes);

  // Check cache
  if (changedNoteIds.length === 0) {
    const cached = readCachedClusters();
    if (cached.length > 0) return cached;
  }

  let clusters: ClusterNode[];

  if (useIterativeRefinement) {
    // Use iterative refinement with memory feedback
    console.log("Starting iterative clustering with memory feedback...");
    const result = await iterativeClusterWithRefinement(notes, 3);
    clusters = result.clusters;
    console.log(
      `Clustering converged: ${result.converged} after ${result.iterations} iterations`
    );
  } else {
    // Use basic dual-prompt clustering
    const result = await dualPromptClusterNotesWithGemini(notes, memoryBuffer);
    clusters = result.clusters;
  }

  writeCachedClusters(clusters);
  return clusters;
};

// Planned: embedding generation + vector-store upserts.
// Stubbed for now; kept to define API for Developer 2/DB task.
export interface EmbeddingChunk {
  noteId: string;
  text: string;
  chunkIndex: number;
}

export const chunkNote = (
  note: Note,
  maxLen = 800,
  overlap = 100
): EmbeddingChunk[] => {
  const chunks: EmbeddingChunk[] = [];
  const text = `${note.title}\n\n${note.content}`;
  let i = 0;
  let idx = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + maxLen);
    const slice = text.slice(i, end);
    chunks.push({ noteId: note.id, text: slice, chunkIndex: idx++ });
    if (end >= text.length) break;
    i = end - overlap;
  }
  return chunks;
};

export const upsertEmbeddings = async (
  _chunks: EmbeddingChunk[]
): Promise<void> => {
  // TODO: Generate embeddings and upsert to vector store with metadata.
  // This will be implemented by Developer 2 / DB task.
};

export const clusterWithEmbeddings = async (
  notes: Note[]
): Promise<ClusterNode[]> => {
  // TODO: Use embeddings + hierarchical clustering.
  // Temporary fallback to LLM clustering.
  return incrementalCluster(notes);
};

// --- Benchmarking utilities ---
export interface BenchmarkResult {
  noteCount: number;
  durationMs: number;
  usedCache: boolean;
}

export const generateSyntheticNotes = (count = 1000): Note[] => {
  const notes: Note[] = [];
  const now = new Date().toISOString().split("T")[0];
  for (let i = 0; i < count; i++) {
    notes.push({
      id: `synthetic-${i}`,
      title: `Transformer Note ${i}`,
      content: `# Attention & Transformers\n\nThis is synthetic content about transformer architecture, attention mechanisms, and embeddings. Index: ${i}.\n\nKey terms: attention, transformer, GPT, BERT, tokens, sequence, context.`,
      tags: ["ai", "nlp", "transformer"],
      createdAt: now,
      folder: "/synthetic/ai",
    });
  }
  return notes;
};

export const benchmarkClustering = async (
  count = 1000
): Promise<BenchmarkResult> => {
  const synthetic = generateSyntheticNotes(count);
  const start = performance.now();
  const before = readCachedClusters();
  const clusters = await incrementalCluster(synthetic);
  const end = performance.now();
  const after = clusters;
  return {
    noteCount: count,
    durationMs: Math.round(end - start),
    usedCache:
      before.length > 0 &&
      synthetic.every((n) => readHashIndex()[n.id]) &&
      after.length === before.length,
  };
};
// --- Hybrid Clustering with Embeddings (Phase 2) ---

/**
 * Convert embedding partitions to ClusterNode structure
 */
const partitionsToClusterNodes = (
  partitions: EmbeddingPartition[],
  notes: Note[]
): ClusterNode[] => {
  return partitions.map((partition) => {
    const noteMap = new Map(notes.map((n) => [n.id, n]));
    const clusterName = `Cluster ${partition.id}`;

    return {
      id: partition.id,
      name: clusterName,
      type: "cluster" as const,
      description: `Embedding-based partition with ${partition.noteIds.length} notes`,
      children: partition.noteIds
        .map((noteId) => {
          const note = noteMap.get(noteId);
          if (!note) return null;
          return {
            id: `${partition.id}-${noteId}`,
            name: note.title,
            type: "note" as const,
            noteId,
          };
        })
        .filter((n): n is ClusterNode => n !== null),
    };
  });
};

/**
 * Hybrid clustering: embeddings for partitioning, then LLM for refinement
 * @param notes Notes to cluster
 * @param similarityThreshold Threshold for embedding-based grouping (0-1)
 * @param useRefinement Whether to refine with dual-prompt LLM
 * @returns Refined clusters
 */
export const hybridClusterWithEmbeddings = async (
  notes: Note[],
  similarityThreshold: number = 0.65,
  useRefinement: boolean = true
): Promise<ClusterNode[]> => {
  console.log(`Starting hybrid clustering for ${notes.length} notes...`);
  const start = performance.now();

  try {
    // Step 1: Generate or load embeddings
    console.log("Generating embeddings...");
    const embeddings = await getOrGenerateEmbeddings(notes);
    const embeddingStats = getEmbeddingStats(embeddings);
    console.log(`Embeddings ready: ${JSON.stringify(embeddingStats)}`);

    if (embeddings.length === 0) {
      console.warn(
        "Failed to generate embeddings, falling back to basic clustering"
      );
      return await enhancedCluster(notes, false);
    }

    // Step 2: Embedding-guided partitioning
    console.log(
      `Performing embedding-guided partitioning (threshold: ${similarityThreshold})...`
    );
    const partitions = embeddingGuidedPartitioning(
      embeddings,
      similarityThreshold,
      2 // minClusterSize
    );

    // Convert to ClusterNodes
    let clusters = partitionsToClusterNodes(partitions, notes);

    // Step 3: Optional LLM refinement for quality
    if (useRefinement && partitions.length > 1) {
      console.log(`Refining ${clusters.length} partitions with LLM...`);

      // For now, enhance the cluster descriptions with LLM
      // Future: iterative refinement of partition assignments
      const refined = await enhancedCluster(notes, false);

      // Merge: use embedding structure but enhance with LLM insights
      if (refined.length > 0) {
        console.log(
          `LLM refinement added insights to ${refined.length} clusters`
        );
        // Merge descriptions from LLM clustering
        clusters.forEach((cluster, i) => {
          if (refined[i]) {
            cluster.description = `${cluster.description} | LLM: ${
              refined[i].description || ""
            }`;
          }
        });
      }
    }

    const duration = Math.round(performance.now() - start);
    console.log(
      `Hybrid clustering complete: ${clusters.length} clusters in ${duration}ms`
    );

    // Cache the result
    writeCachedClusters(clusters);

    return clusters;
  } catch (e) {
    console.error("Hybrid clustering error:", e);
    console.log("Falling back to enhanced clustering...");
    return await enhancedCluster(notes, true);
  }
};

/**
 * Multi-level hybrid clustering for scalability (1000+ notes)
 * @param notes Notes to cluster
 * @param initialThreshold Embedding similarity threshold
 * @param maxClusterSize Maximum notes per cluster before subdivision
 * @returns Hierarchical cluster structure
 */
export const hierarchicalHybridClustering = async (
  notes: Note[],
  initialThreshold: number = 0.65,
  maxClusterSize: number = 200
): Promise<ClusterNode[]> => {
  console.log(
    `Starting hierarchical hybrid clustering for ${notes.length} notes...`
  );

  // Step 1: Initial embedding-based partitioning
  const clusters = await hybridClusterWithEmbeddings(
    notes,
    initialThreshold,
    false
  );

  // Step 2: Subdivide large clusters recursively
  const processCluster = async (
    cluster: ClusterNode,
    depth: number = 1
  ): Promise<ClusterNode> => {
    if (cluster.type === "note" || !cluster.children) {
      return cluster;
    }

    const noteIds = cluster.children
      .filter((c) => c.type === "note")
      .map((c) => c.noteId)
      .filter((id): id is string => Boolean(id));

    // If cluster is too large, subdivide
    if (noteIds.length > maxClusterSize && depth < 3) {
      const childNotes = notes.filter((n) => noteIds.includes(n.id));
      const threshold = initialThreshold - depth * 0.05; // Lower threshold for finer granularity

      console.log(
        `Subdividing cluster "${cluster.name}" (${noteIds.length} notes, depth ${depth})...`
      );

      const subClusters = await hybridClusterWithEmbeddings(
        childNotes,
        Math.max(0.5, threshold),
        false
      );

      return {
        ...cluster,
        children: subClusters,
      };
    }

    return cluster;
  };

  // Process all clusters
  const hierarchical = await Promise.all(
    clusters.map((c) => processCluster(c))
  );

  console.log(
    `Hierarchical clustering complete with ${hierarchical.length} root clusters`
  );
  return hierarchical;
};
// --- Semantic Enhancement (Phase 3) ---

/**
 * Generate semantic centroids for all clusters (Step 3.1)
 * @param clusters Clusters to enhance
 * @param notes All notes for context
 * @returns Map of cluster ID to semantic centroid
 */
export const generateSemanticCentroids = async (
  clusters: ClusterNode[],
  notes: Note[]
): Promise<Map<string, SemanticCentroid>> => {
  const centroids = new Map<string, SemanticCentroid>();
  const noteMap = new Map(notes.map((n) => [n.id, n]));

  for (const cluster of clusters) {
    if (cluster.type === "cluster") {
      // Get notes in this cluster
      const clusterNoteIds =
        cluster.children
          ?.filter((c) => c.type === "note")
          .map((c) => c.noteId)
          .filter((id): id is string => Boolean(id)) || [];

      const clusterNotes = clusterNoteIds
        .map((id) => noteMap.get(id))
        .filter((n): n is Note => Boolean(n));

      if (clusterNotes.length > 0) {
        const centroid = await generateSemanticCentroid(
          cluster.id,
          clusterNotes,
          cluster.description
        );
        centroids.set(cluster.id, centroid);
      }
    }
  }

  console.log(`Generated ${centroids.size} semantic centroids`);
  return centroids;
};

/**
 * Detect hard samples and augment their content (Step 3.2)
 * @param clusters Clusters for context
 * @param notes All notes
 * @param augmentThreshold Threshold for which samples to augment (0-1)
 * @returns Array of hard samples with potential augmentations
 */
export const detectAndAugmentHardSamples = async (
  clusters: ClusterNode[],
  notes: Note[],
  augmentThreshold: number = 0.65
): Promise<HardSample[]> => {
  const hardSamples: HardSample[] = [];
  const clusterIds = clusters
    .filter((c) => c.type === "cluster")
    .map((c) => c.id);

  // Sample a subset of notes to check (for efficiency)
  const samplesToCheck = Math.min(notes.length, 20);
  const sampleIndices = Array.from({ length: samplesToCheck }, (_, i) =>
    Math.floor((i * notes.length) / samplesToCheck)
  );

  for (const idx of sampleIndices) {
    const note = notes[idx];
    if (!note) continue;

    const hardSample = await detectHardSample(note, clusterIds);
    if (hardSample && hardSample.ambiguityScore >= augmentThreshold) {
      hardSamples.push(hardSample);
    }
  }

  console.log(
    `Detected ${hardSamples.length} hard samples (threshold: ${augmentThreshold})`
  );
  return hardSamples;
};

/**
 * Generate supervision signals (constraints) from hard samples (Step 3.3)
 * @param hardSamples Hard samples that need constraints
 * @returns Constraint set with must-link and cannot-link pairs
 */
export const generateSupervisionSignals = (
  hardSamples: HardSample[]
): ConstraintSet => {
  const mustLinkPairs: SupervisionSignal[] = [];
  const cannotLinkPairs: SupervisionSignal[] = [];
  const affectedNotes = new Set<string>();

  // For each hard sample, create constraints
  hardSamples.forEach((sample) => {
    affectedNotes.add(sample.noteId);

    // Must-link: within primary cluster
    if (sample.possibleClusters.length > 0) {
      const primaryCluster = sample.possibleClusters[0];

      // Can-link pairs with other notes from possible clusters
      sample.possibleClusters.forEach((clusterId) => {
        const signal: SupervisionSignal = {
          id: `must-${sample.noteId}-${clusterId}`,
          type: "must-link",
          note1Id: sample.noteId,
          note2Id: `cluster-${clusterId}`, // Pseudo-ID for cluster
          strength: 0.7,
          reason: "Hard sample likely belongs to this cluster",
        };
        mustLinkPairs.push(signal);
      });

      // Cannot-link: away from other clusters
      const otherClusters = sample.possibleClusters.slice(1);
      otherClusters.forEach((clusterId) => {
        const signal: SupervisionSignal = {
          id: `cannot-${sample.noteId}-${clusterId}`,
          type: "cannot-link",
          note1Id: sample.noteId,
          note2Id: `cluster-${clusterId}`,
          strength: 0.3,
          reason: "Hard sample less likely in this cluster",
        };
        cannotLinkPairs.push(signal);
      });
    }
  });

  const result: ConstraintSet = {
    mustLinkPairs,
    cannotLinkPairs,
    totalConstraints: mustLinkPairs.length + cannotLinkPairs.length,
    totalNotesAffected: affectedNotes.size,
  };

  console.log(
    `Generated ${result.totalConstraints} constraints affecting ${result.totalNotesAffected} notes`
  );
  return result;
};

/**
 * Full semantic enhancement pipeline (Phase 3 integration)
 * @param clusters Initial clusters
 * @param notes All notes
 * @param generateCentroids Whether to generate semantic centroids
 * @param detectHardSamples Whether to detect hard samples
 * @returns Enhanced clustering result with semantics
 */
export const semanticEnhancedClustering = async (
  clusters: ClusterNode[],
  notes: Note[],
  generateCentroids: boolean = true,
  detectHardSamples: boolean = true
): Promise<SemanticClusteringResult> => {
  const start = performance.now();

  console.log(
    `Starting semantic enhancement: ${clusters.length} clusters, ${notes.length} notes`
  );

  // Step 1: Generate semantic centroids
  const centroids = generateCentroids
    ? await generateSemanticCentroids(clusters, notes)
    : new Map();

  // Step 2: Detect hard samples
  const hardSamples = detectHardSamples
    ? await detectAndAugmentHardSamples(clusters, notes, 0.65)
    : [];

  // Step 3: Generate supervision signals
  const constraints = generateSupervisionSignals(hardSamples);

  // Step 4: Calculate final confidence
  const avgCentroidConfidence =
    centroids.size > 0
      ? Array.from(centroids.values()).reduce(
          (sum, c) => sum + c.confidence,
          0
        ) / centroids.size
      : 0.8;

  const duration = Math.round(performance.now() - start);

  const result: SemanticClusteringResult = {
    clusters,
    centroids,
    hardSamples,
    constraints,
    iterations: 1,
    finalConfidence: avgCentroidConfidence,
  };

  console.log(
    `Semantic enhancement complete: ${centroids.size} centroids, ${hardSamples.length} hard samples, ${duration}ms`
  );

  return result;
};

/**
 * Full end-to-end semantic clustering: embeddings + LLM + semantics
 * @param notes Notes to cluster
 * @param options Configuration options
 * @returns Complete clustering result with all enhancements
 */
export const fullSemanticClustering = async (
  notes: Note[],
  options: {
    useHybridEmbeddings?: boolean;
    useSemanticEnhancement?: boolean;
    generateCentroids?: boolean;
    detectHardSamples?: boolean;
  } = {}
): Promise<SemanticClusteringResult> => {
  const {
    useHybridEmbeddings = true,
    useSemanticEnhancement = true,
    generateCentroids = true,
    detectHardSamples = true,
  } = options;

  console.log("Starting full semantic clustering pipeline...");

  // Phase 2: Hybrid embeddings
  let clusters: ClusterNode[];
  if (useHybridEmbeddings) {
    console.log("Phase 2: Embedding-guided clustering");
    clusters = await hybridClusterWithEmbeddings(notes, 0.65, true);
  } else {
    console.log("Phase 1: Enhanced LLM clustering");
    clusters = await enhancedCluster(notes, true);
  }

  // Phase 3: Semantic enhancement
  if (useSemanticEnhancement) {
    console.log("Phase 3: Semantic enhancement");
    return semanticEnhancedClustering(
      clusters,
      notes,
      generateCentroids,
      detectHardSamples
    );
  }

  // Fallback: just return clusters
  return {
    clusters,
    centroids: new Map(),
    hardSamples: [],
    constraints: {
      mustLinkPairs: [],
      cannotLinkPairs: [],
      totalConstraints: 0,
      totalNotesAffected: 0,
    },
    iterations: 1,
    finalConfidence: 0.8,
  };
};
