export interface Note {
  id: string;
  title: string;
  content: string; // Markdown content
  tags: string[];
  createdAt: string;
  folder: string; // Virtual folder path
}

export enum SearchMode {
  EXACT = "EXACT", // Grep-style
  SEMANTIC = "SEMANTIC", // LLM-based understanding
  HYBRID = "HYBRID", // Both
}

export interface ClusterNode {
  id: string;
  name: string; // Cluster name or Note title
  type: "cluster" | "note";
  children?: ClusterNode[];
  noteId?: string; // If type is note
  description?: string; // Why this cluster exists
}

export interface SearchResult {
  note: Note;
  score: number;
  reason?: string; // Why matched (for semantic search)
  highlight?: string; // Snippet
}

export interface ProcessingStatus {
  isProcessing: boolean;
  message: string;
}

// --- Memory-Augmented Clustering Types ---

export interface ClusteringDecision {
  id: string;
  timestamp: number;
  noteIds: string[];
  clusterAssignments: Record<string, string>; // noteId -> clusterId
  clusterDescriptions: Record<string, string>; // clusterId -> description
  confidence: number; // 0-1, how confident the clustering was
  method: "initial" | "refinement" | "incremental";
}

export interface ClusteringMemory {
  decisions: ClusteringDecision[];
  lastClusteringTime: number;
  totalNotesProcessed: number;
  averageConfidence: number;
  version: string;
}

export interface MemoryBuffer {
  addDecision(decision: ClusteringDecision): void;
  getRecentDecisions(limit?: number): ClusteringDecision[];
  getDecisionById(id: string): ClusteringDecision | null;
  getMemoryForNotes(noteIds: string[]): ClusteringMemory;
  clearOldDecisions(keepLastN?: number): void;
  getStats(): {
    totalDecisions: number;
    avgConfidence: number;
    lastUpdate: number;
  };
}
