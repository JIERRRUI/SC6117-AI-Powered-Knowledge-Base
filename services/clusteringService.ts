import {
  Note,
  ClusterNode,
  ClusteringDecision,
  ClusteringMemory,
  MemoryBuffer,
} from "../types";
import { clusterNotesWithGemini } from "./geminiService";

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
