/**
 * Clustering Service
 *
 * This service provides intelligent clustering of notes using a combination of:
 * 1. LLM-based semantic analysis (Gemini AI)
 * 2. Embedding-based similarity clustering
 * 3. Hierarchical organization (domains → subtopics → notes)
 *
 * Key Features:
 * - Incremental clustering: Only processes new/changed notes
 * - Hash-based change detection: Efficiently identifies modified notes
 * - Memory buffer: Maintains clustering history for improved consistency
 * - 3-level hierarchy: Root → Domains → Subtopics → Notes
 *
 * Main Entry Points:
 * - incrementalCluster(): Primary function for clustering with smart caching
 * - fullSemanticClustering(): Complete pipeline with semantic enhancement
 */

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
  generateClusterName,
  generateSemanticClusterName,
} from "./geminiService";
import {
  getOrGenerateEmbeddings,
  embeddingGuidedPartitioning,
  getEmbeddingStats,
  loadEmbeddingIndex,
} from "./embeddingService";

// ============================================================================
// Configuration & Constants
// ============================================================================

export interface IngestionResult {
  notes: Note[];
  changedNoteIds: string[];
}

/** Simple content hash for change detection (non-cryptographic) */
const fastHash = (s: string) => {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h.toString(16);
};

// LocalStorage keys for caching
const CACHE_KEY = "clusters_cache_v1";
const HASH_INDEX_KEY = "note_hash_index_v1";
const MEMORY_KEY = "clustering_memory_v1";
const MAX_MEMORY_SIZE = 50; // Keep last 50 clustering decisions

// ============================================================================
// Dynamic Memory System
// ============================================================================

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

  console.log(
    `📊 Ingesting ${notes.length} notes, hash index has ${
      Object.keys(idx).length
    } entries`
  );

  for (const n of notes) {
    const h = fastHash(`${n.title}\n${n.content}`);
    if (idx[n.id] !== h) {
      if (idx[n.id]) {
        console.log(
          `  🔄 Note changed: ${n.title} (${n.id.substring(0, 8)}...)`
        );
      } else {
        console.log(`  🆕 New note: ${n.title} (${n.id.substring(0, 8)}...)`);
      }
      idx[n.id] = h;
      changed.push(n.id);
    }
  }

  writeHashIndex(idx);
  console.log(
    `✅ Detected ${changed.length} changed/new notes out of ${notes.length} total`
  );
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

// Incremental clustering: if few notes changed, only process those and merge into existing clusters
export const incrementalCluster = async (
  notes: Note[],
  existingClusters?: ClusterNode[]
): Promise<ClusterNode[]> => {
  const { changedNoteIds } = ingestNotes(notes);

  // Use passed clusters or fall back to cache
  const cached =
    existingClusters && existingClusters.length > 0
      ? existingClusters
      : readCachedClusters();

  console.log(
    `  📦 Using ${cached.length} existing clusters ${
      existingClusters ? "(from React state)" : "(from cache)"
    }`
  );

  // No changes - return cached clusters
  if (changedNoteIds.length === 0) {
    if (cached.length > 0) {
      console.log(
        `✅ No changes detected, using cached ${cached.length} clusters`
      );
      return cached;
    }
  }

  // True incremental: only process new/changed notes if we have existing clusters
  // and changes are less than 30% of total notes
  const changeRatio = changedNoteIds.length / notes.length;
  if (cached.length > 0 && changeRatio < 0.3) {
    console.log(
      `🔄 Incremental update: ${
        changedNoteIds.length
      } changed notes (${Math.round(changeRatio * 100)}%)`
    );

    // Get only the new/changed notes
    const changedNotes = notes.filter((n) => changedNoteIds.includes(n.id));

    // Load ALL existing embeddings from cache FIRST
    const embeddingIndex = loadEmbeddingIndex();
    let allEmbeddings = [...embeddingIndex.embeddings];
    console.log(`  📊 Loaded ${allEmbeddings.length} cached embeddings`);

    // Generate embeddings for new notes only (using the full notes list to preserve cache)
    // Pass ALL notes but only new ones will be generated due to cache check
    const updatedEmbeddings = await getOrGenerateEmbeddings(notes, false);
    allEmbeddings = updatedEmbeddings;

    // Get embeddings for just the new notes
    const newEmbeddings = allEmbeddings.filter((e) =>
      changedNotes.some((n) => n.id === e.noteId)
    );

    console.log(`  📊 Total embeddings available: ${allEmbeddings.length}`);
    console.log(
      `  🆕 New embeddings for changed notes: ${newEmbeddings.length}`
    );

    // True incremental update - add new notes to existing clusters
    const updatedClusters = [...cached];

    for (const newNote of changedNotes) {
      // Check if note already exists in clusters (avoid duplicates)
      const alreadyExists = updatedClusters.some((cluster) =>
        cluster.children?.some((c) => c.noteId === newNote.id)
      );
      if (alreadyExists) {
        console.log(
          `  ⏭️ Note "${newNote.title}" already in clusters, skipping`
        );
        continue;
      }

      const newEmb = newEmbeddings.find((e) => e.noteId === newNote.id);

      // Use embedding similarity matching if we have the new note's embedding
      if (newEmb && allEmbeddings.length > 1) {
        // Find most similar existing cluster
        let bestCluster: ClusterNode | null = null;
        let bestSimilarity = 0;
        let bestClusterName = "";

        for (const cluster of updatedClusters) {
          if (cluster.type !== "cluster" || !cluster.children) continue;

          // Get embeddings of notes in this cluster (recursively check nested clusters too)
          const clusterNoteIds = collectNoteIds(cluster);

          const clusterEmbeddings = allEmbeddings.filter((e) =>
            clusterNoteIds.includes(e.noteId)
          );

          if (clusterEmbeddings.length === 0) continue;

          // Compute average similarity to cluster
          let totalSim = 0;
          for (const cEmb of clusterEmbeddings) {
            const sim = cosineSimilarity(newEmb.vector, cEmb.vector);
            totalSim += sim;
          }
          const avgSim = totalSim / clusterEmbeddings.length;

          if (avgSim > bestSimilarity) {
            bestSimilarity = avgSim;
            bestCluster = cluster;
            bestClusterName = cluster.name;
          }
        }

        // Add to best matching cluster if similarity > 0.5 (higher threshold for embeddings)
        if (bestCluster && bestSimilarity > 0.5) {
          console.log(
            `  ➕ Adding "${
              newNote.title
            }" to cluster "${bestClusterName}" (embedding similarity: ${bestSimilarity.toFixed(
              2
            )})`
          );
          bestCluster.children!.push({
            id: `note-${newNote.id}`,
            name: newNote.title,
            type: "note",
            noteId: newNote.id,
          });
          continue;
        }
      }

      // Fallback: Use content-based matching to find best cluster
      const matchedCluster = await findBestClusterForNote(
        newNote,
        updatedClusters
      );

      if (matchedCluster) {
        console.log(
          `  🎯 Adding "${newNote.title}" to cluster "${matchedCluster.name}" (content match)`
        );
        matchedCluster.children = matchedCluster.children || [];
        matchedCluster.children.push({
          id: `note-${newNote.id}`,
          name: newNote.title,
          type: "note",
          noteId: newNote.id,
        });
      } else {
        // Create a new cluster with a meaningful name based on note content
        const clusterName = await generateClusterNameForNote(newNote);
        console.log(
          `  🆕 Creating new cluster "${clusterName}" for "${newNote.title}"`
        );
        updatedClusters.push({
          id: `cluster-new-${Date.now()}-${newNote.id}`,
          name: clusterName,
          type: "cluster",
          description: `Cluster created for: ${newNote.title}`,
          children: [
            {
              id: `note-${newNote.id}`,
              name: newNote.title,
              type: "note",
              noteId: newNote.id,
            },
          ],
        });
      }
    }

    writeCachedClusters(updatedClusters);
    console.log(
      `✅ Incremental clustering complete: ${updatedClusters.length} clusters`
    );
    return updatedClusters;
  }

  // Major changes or no existing clusters - run full clustering
  console.log(
    `🚀 Running full semantic clustering pipeline (${
      changedNoteIds.length
    } changes, ${Math.round(changeRatio * 100)}%)...`
  );
  const result = await fullSemanticClustering(notes, {
    useHybridEmbeddings: true,
    useSemanticEnhancement: true,
    generateCentroids: true,
    detectHardSamples: true,
  });

  console.log(
    `✅ Clustering complete: ${
      result.clusters.length
    } clusters, confidence ${result.finalConfidence.toFixed(2)}`
  );
  console.log(
    `📊 Centroids: ${result.centroids.size}, Hard samples: ${result.hardSamples.length}, Constraints: ${result.constraints.totalConstraints}`
  );

  writeCachedClusters(result.clusters);
  return result.clusters;
};

// Helper: Compute cosine similarity between two vectors (used in incremental clustering)
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

// Helper: Recursively collect all note IDs from a cluster (including nested clusters)
const collectNoteIds = (cluster: ClusterNode): string[] => {
  const noteIds: string[] = [];
  if (!cluster.children) return noteIds;

  for (const child of cluster.children) {
    if (child.type === "note" && child.noteId) {
      noteIds.push(child.noteId);
    } else if (child.type === "cluster") {
      noteIds.push(...collectNoteIds(child));
    }
  }
  return noteIds;
};

// Helper: Extract keywords from text for matching
const extractKeywords = (text: string): string[] => {
  const stopWords = new Set([
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "it",
    "its",
  ]);

  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !stopWords.has(word));
};

// Helper: Find best matching cluster for a note using content analysis
const findBestClusterForNote = async (
  note: Note,
  clusters: ClusterNode[]
): Promise<ClusterNode | null> => {
  const noteKeywords = extractKeywords(note.title + " " + note.content);

  if (noteKeywords.length === 0) return null;

  let bestCluster: ClusterNode | null = null;
  let bestScore = 0;
  let bestMatchDetails = "";

  for (const cluster of clusters) {
    if (cluster.type !== "cluster") continue;

    // Build cluster keywords from name and child note names
    const clusterText = [
      cluster.name,
      cluster.description || "",
      ...(cluster.children?.map((c) => c.name) || []),
    ].join(" ");

    const clusterKeywords = extractKeywords(clusterText);

    // Calculate keyword overlap - STRICT matching only (exact match or one contains the other with min 4 chars)
    const overlap = noteKeywords.filter((kw) =>
      clusterKeywords.some((ck) => {
        // Exact match
        if (kw === ck) return true;
        // One contains the other (only for words >= 4 chars to avoid false positives)
        if (kw.length >= 4 && ck.length >= 4) {
          if (ck.includes(kw) || kw.includes(ck)) return true;
        }
        return false;
      })
    );

    // Require at least 40% overlap and at least 1 matching keyword
    const score = overlap.length / Math.max(noteKeywords.length, 1);

    if (overlap.length >= 1 && score > bestScore && score >= 0.4) {
      bestScore = score;
      bestCluster = cluster;
      bestMatchDetails = `matched keywords: [${overlap.join(", ")}]`;
    }
  }

  if (bestCluster) {
    console.log(
      `    📝 Content match: ${bestMatchDetails} (score: ${bestScore.toFixed(
        2
      )})`
    );
  }

  return bestCluster;
};

// Helper: Levenshtein distance for fuzzy matching
const levenshteinDistance = (str1: string, str2: string): number => {
  const m = str1.length;
  const n = str2.length;
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (str1[i - 1] === str2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  return dp[m][n];
};

// Helper: Generate a meaningful cluster name for a new note
const generateClusterNameForNote = async (note: Note): Promise<string> => {
  // Try to derive from note title first
  const titleWords = extractKeywords(note.title);

  if (titleWords.length > 0) {
    // Create a topic name from first 2-3 meaningful words
    const topicWords = titleWords.slice(0, 3);
    return topicWords
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  }

  // Fallback to content analysis
  const contentWords = extractKeywords(note.content);
  if (contentWords.length > 0) {
    // Find most frequent words
    const wordFreq: Record<string, number> = {};
    for (const word of contentWords) {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    }

    const topWords = Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([word]) => word.charAt(0).toUpperCase() + word.slice(1));

    if (topWords.length > 0) {
      return topWords.join(" ");
    }
  }

  // Last resort
  return "Miscellaneous";
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

// --- Hybrid Clustering with Embeddings (Phase 2) ---

/**
 * Convert embedding partitions to ClusterNode structure
 * @param partitions Embedding-based partitions
 * @param notes All original notes
 * @param partitionNames Optional LLM-generated names and descriptions for each partition
 */
const partitionsToClusterNodes = (
  partitions: EmbeddingPartition[],
  notes: Note[],
  partitionNames?: Array<{ name: string; description: string }>
): ClusterNode[] => {
  const noteMap = new Map(notes.map((n) => [n.id, n]));

  // Track which notes have been clustered
  const clusteredNoteIds = new Set<string>();

  const clusters = partitions.map((partition, index) => {
    // Use LLM-generated name if available
    let clusterName = `Cluster ${index + 1}`;
    let clusterDescription = `Partition with ${partition.noteIds.length} notes`;

    if (partitionNames && partitionNames[index]) {
      clusterName = partitionNames[index].name;
      clusterDescription = partitionNames[index].description;
    }

    partition.noteIds.forEach((id) => clusteredNoteIds.add(id));

    return {
      id: partition.id,
      name: clusterName,
      type: "cluster" as const,
      description: clusterDescription,
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
        .filter(
          (
            n
          ): n is { id: string; name: string; type: "note"; noteId: string } =>
            n !== null
        ),
    };
  });

  // Add unclustered notes - they will get proper names in Phase 3
  const unclusteredNotes = notes.filter((n) => !clusteredNoteIds.has(n.id));
  if (unclusteredNotes.length > 0) {
    console.log(
      `⚠️ Found ${unclusteredNotes.length} unclustered notes, creating additional cluster`
    );

    // Create a temporary cluster - Phase 3 will give it a proper name
    clusters.push({
      id: "unclustered-group",
      name: "Additional Topics", // Will be renamed by LLM in Phase 3
      type: "cluster" as const,
      description: `Notes requiring semantic analysis (${unclusteredNotes.length})`,
      children: unclusteredNotes.map((note) => ({
        id: `unclustered-${note.id}`,
        name: note.title,
        type: "note" as const,
        noteId: note.id,
      })),
    });
  }

  return clusters;
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
  similarityThreshold: number = 0.4,
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
      1 // minClusterSize - allow more granular clusters
    );

    // Step 3: Generate meaningful names for each partition using LLM
    console.log(
      `Generating meaningful names for ${partitions.length} partitions...`
    );
    const partitionNames = await Promise.all(
      partitions.map(async (partition) => {
        const partitionNotes = notes.filter((n) =>
          partition.noteIds.includes(n.id)
        );
        if (!useRefinement || partitionNotes.length === 0) {
          return {
            name: `Topic Group`,
            description: `Group of ${partition.noteIds.length} notes`,
          };
        }
        try {
          return await generateClusterName(partitionNotes);
        } catch (e) {
          console.error(
            `Failed to generate name for partition ${partition.id}:`,
            e
          );
          return {
            name: `Topic Group`,
            description: `Group of ${partition.noteIds.length} notes`,
          };
        }
      })
    );
    console.log(`Generated ${partitionNames.length} cluster names`);

    // Convert to ClusterNodes with meaningful names
    let clusters = partitionsToClusterNodes(partitions, notes, partitionNames);

    // FORCE SPLIT: If we only got 1 cluster with many notes, force split it
    if (clusters.length === 1 && notes.length >= 10) {
      console.warn(
        `⚠️ Only 1 cluster detected with ${notes.length} notes. Forcing split by folder...`
      );

      // Group by folder as fallback
      const folderGroups = new Map<string, Note[]>();
      notes.forEach((note) => {
        const folder = note.folder || "/misc";
        if (!folderGroups.has(folder)) folderGroups.set(folder, []);
        folderGroups.get(folder)!.push(note);
      });

      // Create clusters from folders
      const folderClusters: ClusterNode[] = [];
      let idx = 0;
      for (const [folder, folderNotes] of folderGroups) {
        if (folderNotes.length > 0) {
          folderClusters.push({
            id: `folder-cluster-${idx++}`,
            name: folder
              .replace(/^\//, "")
              .replace(/-/g, " ")
              .replace(/\b\w/g, (l) => l.toUpperCase()),
            type: "cluster",
            description: `Notes from ${folder}`,
            children: folderNotes.map((note) => ({
              id: `note-${note.id}`,
              name: note.title,
              type: "note" as const,
              noteId: note.id,
            })),
          });
        }
      }

      clusters = folderClusters;
      console.log(
        `✅ Force-split into ${clusters.length} folder-based clusters`
      );
    }

    const duration = Math.round(performance.now() - start);
    //print for debugging
    console.log("Clusters after embedding partitioning and naming:", clusters);
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
 * 3-Level Hierarchical Clustering:
 * Level 0: Root (single node containing all)
 * Level 1: High-level domains (e.g., "Technology", "Cooking", "Personal")
 * Level 2: Subtopics (e.g., "Machine Learning", "Web Development")
 *
 * @param notes Notes to cluster
 * @param domainThreshold Threshold for Level 1 domains (higher = fewer, broader domains)
 * @param subtopicThreshold Threshold for Level 2 subtopics (lower = more granular subtopics)
 * @returns Hierarchical cluster structure with root node
 */
export const hierarchicalHybridClustering = async (
  notes: Note[],
  domainThreshold: number = 0.75,
  subtopicThreshold: number = 0.8
): Promise<ClusterNode[]> => {
  console.log(
    `🏗️ Starting 3-level hierarchical clustering for ${notes.length} notes...`
  );
  console.log(
    `   Domain threshold: ${domainThreshold}, Subtopic threshold: ${subtopicThreshold}`
  );

  // Step 1: Create Level 1 - High-level domains (broad grouping)
  console.log("📊 Level 1: Creating high-level domains...");
  const domains = await hybridClusterWithEmbeddings(
    notes,
    domainThreshold,
    true // Generate names for domains
  );
  console.log(`   Created ${domains.length} high-level domains`);

  // Step 2: For each domain, create Level 2 - Subtopics (finer grouping)
  console.log("📂 Level 2: Creating subtopics within each domain...");
  const domainsWithSubtopics = await Promise.all(
    domains.map(async (domain) => {
      // Get notes in this domain
      const domainNoteIds =
        domain.children
          ?.filter((c) => c.type === "note")
          .map((c) => c.noteId)
          .filter((id): id is string => Boolean(id)) || [];

      // If domain has enough notes, subdivide into subtopics
      if (domainNoteIds.length >= 3) {
        const domainNotes = notes.filter((n) => domainNoteIds.includes(n.id));

        console.log(
          `   Subdividing "${domain.name}" (${domainNotes.length} notes)...`
        );

        try {
          const subtopics = await hybridClusterWithEmbeddings(
            domainNotes,
            subtopicThreshold,
            true // Generate names for subtopics
          );

          // If we got meaningful subtopics (more than 1), use them
          if (subtopics.length > 1) {
            console.log(`     → ${subtopics.length} subtopics created`);
            return {
              ...domain,
              children: subtopics, // Replace flat notes with subtopic clusters
            };
          }
        } catch (e) {
          console.warn(
            `     → Subtopic creation failed, keeping flat structure`
          );
        }
      }

      // Keep original structure if not enough notes or subdivision failed
      return domain;
    })
  );

  // Step 3: Create Level 0 - Root node wrapping everything
  const rootNode: ClusterNode = {
    id: "root",
    name: "Knowledge Base",
    type: "cluster",
    description: `All notes organized into ${domainsWithSubtopics.length} domains`,
    children: domainsWithSubtopics,
  };

  console.log(`✅ Hierarchical clustering complete:`);
  console.log(`   Level 0: 1 root node`);
  console.log(`   Level 1: ${domainsWithSubtopics.length} domains`);
  const totalSubtopics = domainsWithSubtopics.reduce((sum, d) => {
    const subtopicCount =
      d.children?.filter((c) => c.type === "cluster").length || 0;
    return sum + subtopicCount;
  }, 0);
  console.log(`   Level 2: ${totalSubtopics} subtopics`);

  // Return array with root node (or just domains if you prefer flat array)
  // For graph visualization, returning domains might be better
  return domainsWithSubtopics;
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

  // Helper: Recursively get all notes from a cluster
  const getNotesFromCluster = (cluster: ClusterNode): Note[] => {
    const result: Note[] = [];
    if (!cluster.children) return result;

    for (const child of cluster.children) {
      if (child.type === "note" && child.noteId) {
        const note = noteMap.get(child.noteId);
        if (note) result.push(note);
      } else if (child.type === "cluster") {
        result.push(...getNotesFromCluster(child));
      }
    }
    return result;
  };

  // Recursively process clusters and subclusters
  const processCluster = async (cluster: ClusterNode) => {
    if (cluster.type === "cluster") {
      // Get all notes (including from nested subclusters)
      const clusterNotes = getNotesFromCluster(cluster);

      if (clusterNotes.length > 0) {
        const centroid = await generateSemanticCentroid(
          cluster.id,
          clusterNotes,
          cluster.description
        );
        centroids.set(cluster.id, centroid);
      }

      // Process subclusters too
      if (cluster.children) {
        for (const child of cluster.children) {
          if (child.type === "cluster") {
            await processCluster(child);
          }
        }
      }
    }
  };

  console.log("Note Map size:", noteMap.size);
  console.log("Clusters to process:", clusters.length);
  console.log("Generating semantic centroids for clusters...");

  for (const cluster of clusters) {
    await processCluster(cluster);
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

  // Step 3: Re-enhance cluster names with semantic analysis
  console.log("Enhancing cluster names with semantic context...");
  const noteMap = new Map(notes.map((n) => [n.id, n]));

  // Helper: Recursively get all notes from a cluster (including nested subclusters)
  const getClusterNotes = (cluster: ClusterNode): Note[] => {
    const clusterNotes: Note[] = [];
    if (!cluster.children) return clusterNotes;

    for (const child of cluster.children) {
      if (child.type === "note" && child.noteId) {
        const note = noteMap.get(child.noteId);
        if (note) clusterNotes.push(note);
      } else if (child.type === "cluster") {
        // Recursively get notes from subclusters
        clusterNotes.push(...getClusterNotes(child));
      }
    }
    return clusterNotes;
  };

  // Recursively enhance cluster and all its subclusters
  const enhanceClusterRecursively = async (
    cluster: ClusterNode
  ): Promise<ClusterNode> => {
    // First, recursively enhance any subclusters
    let enhancedChildren = cluster.children;
    if (cluster.children) {
      enhancedChildren = await Promise.all(
        cluster.children.map(async (child) => {
          if (child.type === "cluster") {
            return await enhanceClusterRecursively(child);
          }
          return child;
        })
      );
    }

    // Get all notes in this cluster (including from subclusters)
    const clusterNotes = getClusterNotes(cluster);

    if (clusterNotes.length === 0) {
      return { ...cluster, children: enhancedChildren };
    }

    try {
      const centroid = centroids.get(cluster.id);
      const { name, description } = await generateSemanticClusterName(
        clusterNotes,
        centroid
      );

      return {
        ...cluster,
        name,
        description,
        children: enhancedChildren,
      };
    } catch (e) {
      console.error(`Failed to enhance cluster ${cluster.id}:`, e);
      return { ...cluster, children: enhancedChildren };
    }
  };

  const enhancedClusters = await Promise.all(
    clusters.map((cluster) => enhanceClusterRecursively(cluster))
  );

  // Step 4: Generate supervision signals
  const constraints = generateSupervisionSignals(hardSamples);

  // Step 5: Apply constraints (Self-Healing)
  // Logic: Move hard samples to their 'must-link' clusters if specified
  if (constraints.mustLinkPairs.length > 0) {
    console.log("⚡ Applying supervision signals to refine clusters...");

    // Helper to find which cluster currently contains a note
    const findParentCluster = (noteId: string): ClusterNode | undefined => {
      return enhancedClusters.find((c) =>
        c.children?.some((child) => child.noteId === noteId)
      );
    };

    let movedCount = 0;

    // Only process 'must-link' constraints where the target is a cluster
    constraints.mustLinkPairs.forEach((signal) => {
      if (
        signal.type === "must-link" &&
        signal.note2Id.startsWith("cluster-")
      ) {
        const noteId = signal.note1Id;
        const targetClusterId = signal.note2Id.replace("cluster-", "");

        // Find current home and target home
        const currentCluster = findParentCluster(noteId);
        // Match partition ID format "partition-X"
        const targetCluster = enhancedClusters.find(
          (c) =>
            c.id === targetClusterId || c.id === `partition-${targetClusterId}`
        );

        if (
          currentCluster &&
          targetCluster &&
          currentCluster.id !== targetCluster.id
        ) {
          // Find the node in the current cluster
          const noteNodeIndex = currentCluster.children!.findIndex(
            (c) => c.noteId === noteId
          );

          if (noteNodeIndex > -1) {
            // Remove from current
            const [noteNode] = currentCluster.children!.splice(
              noteNodeIndex,
              1
            );

            // Add to target
            if (!targetCluster.children) targetCluster.children = [];
            targetCluster.children.push(noteNode);

            console.log(
              `Self-Healing: Moved note ${noteId} from "${currentCluster.name}" to "${targetCluster.name}"`
            );
            movedCount++;
          }
        }
      }
    });

    if (movedCount > 0) {
      console.log(
        `✅ Self-Healing complete: Moved ${movedCount} notes based on semantic analysis.`
      );
    }
  }

  // Step 6: Calculate final confidence
  const avgCentroidConfidence =
    centroids.size > 0
      ? Array.from(centroids.values()).reduce(
          (sum, c) => sum + c.confidence,
          0
        ) / centroids.size
      : 0.8;

  const duration = Math.round(performance.now() - start);

  const result: SemanticClusteringResult = {
    clusters: enhancedClusters,
    centroids,
    hardSamples,
    constraints,
    iterations: 1,
    finalConfidence: avgCentroidConfidence,
  };

  console.log(
    `Semantic enhancement complete: ${centroids.size} centroids, ${hardSamples.length} hard samples, ${enhancedClusters.length} clusters enhanced, ${duration}ms`
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

  // Update hash index to track these notes for future incremental updates
  ingestNotes(notes);

  // Phase 2: Hybrid embeddings with 3-level hierarchy
  let clusters: ClusterNode[];
  if (useHybridEmbeddings) {
    console.log("Phase 2: 3-Level Hierarchical Clustering");
    // Level 0: Root, Level 1: Domains, Level 2: Subtopics
    clusters = await hierarchicalHybridClustering(
      notes,
      0.3, // Domain threshold (lower = more notes grouped together)
      0.5 // Subtopic threshold (higher = finer subtopics)
    );
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
