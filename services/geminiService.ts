import { ClusteringDecision, MemoryBuffer } from "../types";
import { memoryBuffer as globalMemoryBuffer } from "./clusteringService";
// --- Dual-Prompt Clustering Logic ---
/**
 * Performs dual-prompt clustering: initial rough grouping, then refinement with memory feedback.
 * @param notes Notes to cluster
 * @param memory Optional memory buffer (defaults to global buffer)
 * @returns Refined clusters and clustering decision metadata
 */
export const dualPromptClusterNotesWithGemini = async (
  notes: Note[],
  memory: MemoryBuffer = globalMemoryBuffer
): Promise<{ clusters: ClusterNode[]; decision: ClusteringDecision }> => {
  const ai = getAIClient();
  const notesLite = notes.map((n) => ({
    id: n.id,
    title: n.title,
    contentSnippet: n.content.substring(0, 200),
    tags: n.tags,
  }));

  // 1. Initial prompt for rough grouping and cluster count estimation
  const initialPrompt = `
    You are an expert knowledge manager.
    Analyze the following notes and group them into rough high-level clusters.
    Estimate the optimal number of clusters based on content themes.
    Return a JSON array of clusters, each with a name, description, and assigned note IDs.
    
    Notes:
    ${JSON.stringify(notesLite)}
  `;

  const initialResponse = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: initialPrompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            name: { type: Type.STRING },
            description: { type: Type.STRING },
            noteIds: { type: Type.ARRAY, items: { type: Type.STRING } },
          },
          required: ["id", "name", "description", "noteIds"],
        },
      },
    },
  });

  let initialClusters: Array<{
    id: string;
    name: string;
    description: string;
    noteIds: string[];
  }> = [];
  try {
    initialClusters = JSON.parse(initialResponse.text || "[]");
  } catch (e) {
    console.error("Initial clustering parse error", e);
    initialClusters = [];
  }

  // 2. Use memory buffer for feedback (recent decisions)
  const recentDecisions = memory.getRecentDecisions(3);
  const memorySummary = recentDecisions
    .map(
      (d) =>
        `Decision ${d.id}: ${
          Object.keys(d.clusterDescriptions).length
        } clusters, confidence ${d.confidence}`
    )
    .join("\n");

  // 3. Refinement prompt for optimal assignment
  const refinementPrompt = `
    You are an expert knowledge manager.
    Refine the following initial clusters for optimal grouping and cluster count.
    Use the provided memory of recent clustering decisions as feedback.
    Return a hierarchical cluster tree (JSON) with cluster names, descriptions, and note assignments.
    
    Initial Clusters:
    ${JSON.stringify(initialClusters)}
    
    Memory Feedback:
    ${memorySummary}
    
    Notes:
    ${JSON.stringify(notesLite)}
  `;

  const refinementResponse = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: refinementPrompt,
    config: {
      systemInstruction:
        "You are a helpful assistant that organizes knowledge.",
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            name: { type: Type.STRING },
            type: { type: Type.STRING, enum: ["cluster"] },
            description: { type: Type.STRING },
            children: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  name: { type: Type.STRING },
                  type: { type: Type.STRING, enum: ["note"] },
                  noteId: { type: Type.STRING },
                },
                required: ["id", "name", "type", "noteId"],
              },
            },
          },
          required: ["id", "name", "type", "children"],
        },
      },
    },
  });

  let clusters: ClusterNode[] = [];
  try {
    clusters = JSON.parse(refinementResponse.text || "[]");
  } catch (e) {
    console.error("Refinement clustering parse error", e);
    clusters = [];
  }

  // 4. Build clustering decision metadata
  const now = Date.now();
  const clusterAssignments: Record<string, string> = {};
  const clusterDescriptions: Record<string, string> = {};
  clusters.forEach((cluster) => {
    clusterDescriptions[cluster.id] = cluster.description || "";
    (cluster.children || []).forEach((child) => {
      if (child.noteId) clusterAssignments[child.noteId] = cluster.id;
    });
  });

  const decision: ClusteringDecision = {
    id: `decision-${now}`,
    timestamp: now,
    noteIds: notes.map((n) => n.id),
    clusterAssignments,
    clusterDescriptions,
    confidence: 0.9, // Placeholder, could be estimated from LLM output
    method: "refinement",
  };

  // 5. Persist decision in memory buffer
  memory.addDecision(decision);

  return { clusters, decision };
};
import { GoogleGenAI, Type } from "@google/genai";
import { Note, ClusterNode, SearchResult } from "../types";

const getAIClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// --- Auto-Clustering Logic ---

export const clusterNotesWithGemini = async (
  notes: Note[]
): Promise<ClusterNode[]> => {
  const ai = getAIClient();

  // Prepare a lightweight representation of notes to save tokens
  const notesLite = notes.map((n) => ({
    id: n.id,
    title: n.title,
    contentSnippet: n.content.substring(0, 200), // Only send first 200 chars
    tags: n.tags,
  }));

  const prompt = `
    You are an expert knowledge manager. 
    Analyze the following list of notes and organize them into a hierarchical cluster structure.
    Create high-level categories based on the content themes (e.g., "Artificial Intelligence", "Cooking", "Web Development").
    Assign each note to the most relevant category.
    
    Notes:
    ${JSON.stringify(notesLite)}
  `;

  // We define a schema for the tree structure
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      systemInstruction:
        "You are a helpful assistant that organizes knowledge.",
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            name: { type: Type.STRING },
            type: { type: Type.STRING, enum: ["cluster"] },
            description: { type: Type.STRING },
            children: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  name: { type: Type.STRING }, // Note title
                  type: { type: Type.STRING, enum: ["note"] },
                  noteId: { type: Type.STRING }, // Reference to original note ID
                },
                required: ["id", "name", "type", "noteId"],
              },
            },
          },
          required: ["id", "name", "type", "children"],
        },
      },
    },
  });

  const jsonStr = response.text || "[]";
  try {
    return JSON.parse(jsonStr) as ClusterNode[];
  } catch (e) {
    console.error("Failed to parse clustering result", e);
    return [];
  }
};

// --- Semantic Search Logic ---

export const semanticSearchWithGemini = async (
  query: string,
  notes: Note[]
): Promise<SearchResult[]> => {
  const ai = getAIClient();

  const notesLite = notes.map((n) => ({
    id: n.id,
    title: n.title,
    summary: n.content.substring(0, 300),
  }));

  const prompt = `
    User Query: "${query}"

    Task: Rank the following notes based on their relevance to the user query.
    Return a list of the top relevant notes. 
    For each note, provide a relevance score (0-100) and a brief reasoning.
    If a note is not relevant, do not include it.

    Notes Data:
    ${JSON.stringify(notesLite)}
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            noteId: { type: Type.STRING },
            score: { type: Type.NUMBER },
            reason: { type: Type.STRING },
          },
          required: ["noteId", "score", "reason"],
        },
      },
    },
  });

  const jsonStr = response.text || "[]";
  try {
    const rawResults = JSON.parse(jsonStr) as {
      noteId: string;
      score: number;
      reason: string;
    }[];

    // Merge back with full note objects
    const results: SearchResult[] = rawResults
      .map((r): SearchResult | null => {
        const fullNote = notes.find((n) => n.id === r.noteId);
        if (!fullNote) return null;
        return {
          note: fullNote,
          score: r.score,
          reason: r.reason,
        };
      })
      .filter((r): r is SearchResult => r !== null)
      .sort((a, b) => b.score - a.score);

    return results;
  } catch (e) {
    console.error("Semantic search failed", e);
    return [];
  }
};

// --- Content Correction Logic ---

export const correctTextWithGemini = async (text: string): Promise<string> => {
  const ai = getAIClient();
  const prompt = `
    You are a professional editor.
    Please correct the grammar, spelling, and punctuation of the following text.
    Correct any phonetic misspellings (e.g., "pronouciation" -> "pronunciation").
    Maintain the original markdown formatting.
    Do not add any conversational filler, just return the corrected text.
    
    Text to fix:
    ${text}
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
  });

  return response.text?.trim() || text;
};
// --- Semantic Centroid Generation (Phase 3, Step 3.1) ---

import { SemanticCentroid, HardSample } from "../types";

/**
 * Generate semantic descriptions for cluster centroids
 * @param clusterId Cluster ID
 * @param clusterNotes Notes in the cluster
 * @param description Existing cluster description (optional)
 * @returns Semantic centroid with LLM-generated insights
 */
export const generateSemanticCentroid = async (
  clusterId: string,
  clusterNotes: Note[],
  description?: string
): Promise<SemanticCentroid> => {
  const ai = getAIClient();

  // Prepare cluster summary
  const notesSummary = clusterNotes
    .slice(0, 5) // Limit to first 5 notes for context
    .map((n) => `- ${n.title}: ${n.content.substring(0, 100)}...`)
    .join("\n");

  const prompt = `
    You are a knowledge organizer expert.
    Analyze this cluster of notes and provide a semantic description.
    
    Cluster Notes:
    ${notesSummary}
    
    Current Description: ${description || "None"}
    
    Task:
    1. Provide a clear, concise semantic description (1-2 sentences) of what unifies these notes
    2. Extract 3-5 key terms/keywords that capture the essence of this cluster
    3. Rate your confidence (0-1) in this grouping
    
    Format your response as JSON: { "description": "...", "keywords": [...], "confidence": 0.9 }
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            description: { type: Type.STRING },
            keywords: { type: Type.ARRAY, items: { type: Type.STRING } },
            confidence: { type: Type.NUMBER },
          },
          required: ["description", "keywords", "confidence"],
        },
      },
    });

    const result = JSON.parse(response.text || "{}") as {
      description: string;
      keywords: string[];
      confidence: number;
    };

    const centroid: SemanticCentroid = {
      clusterId,
      description: result.description,
      keywords: result.keywords || [],
      confidence: Math.max(0, Math.min(1, result.confidence || 0.8)),
      generatedAt: Date.now(),
    };

    console.log(
      `Generated semantic centroid for ${clusterId}: ${centroid.description}`
    );
    return centroid;
  } catch (e) {
    console.error("Failed to generate semantic centroid:", e);
    return {
      clusterId,
      description: description || "Cluster",
      keywords: [],
      confidence: 0.5,
      generatedAt: Date.now(),
    };
  }
};

/**
 * Detect ambiguous/hard samples that could belong to multiple clusters
 * @param note Note to analyze
 * @param clusterIds Possible cluster IDs
 * @returns Hard sample analysis
 */
export const detectHardSample = async (
  note: Note,
  clusterIds: string[]
): Promise<HardSample | null> => {
  const ai = getAIClient();

  const prompt = `
    You are a data quality analyzer.
    Analyze this note for ambiguity - does it clearly belong to one category or could it fit multiple?
    
    Note Title: ${note.title}
    Note Content: ${note.content.substring(0, 300)}...
    Possible Clusters: ${clusterIds.join(", ")}
    
    Task:
    1. Rate ambiguity (0-1): 0 = clear assignment, 1 = highly ambiguous
    2. If ambiguous, suggest which clusters it could fit
    3. Suggest a clearer rewrite if needed
    
    Format: { "ambiguityScore": 0.7, "possibleClusters": ["cluster-1"], "shouldAugment": true, "augmentedContent": "..." }
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            ambiguityScore: { type: Type.NUMBER },
            possibleClusters: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            shouldAugment: { type: Type.BOOLEAN },
            augmentedContent: { type: Type.STRING },
          },
          required: ["ambiguityScore", "possibleClusters", "shouldAugment"],
        },
      },
    });

    const result = JSON.parse(response.text || "{}");

    // Only flag as hard sample if ambiguous
    if (result.ambiguityScore > 0.6) {
      const hardSample: HardSample = {
        noteId: note.id,
        originalContent: note.content,
        ambiguityScore: Math.max(0, Math.min(1, result.ambiguityScore || 0.5)),
        possibleClusters: result.possibleClusters || clusterIds,
        augmentedContent: result.augmentedContent,
        rewriteReason: result.shouldAugment ? "High ambiguity" : undefined,
      };

      console.log(
        `Detected hard sample: ${
          note.id
        } (ambiguity: ${hardSample.ambiguityScore.toFixed(2)})`
      );
      return hardSample;
    }

    return null;
  } catch (e) {
    console.error("Failed to detect hard sample:", e);
    return null;
  }
};
