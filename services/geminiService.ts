import { ClusteringDecision, MemoryBuffer } from "../types";
import { memoryBuffer as globalMemoryBuffer } from "./clusteringService";
import { GoogleGenAI, Type } from "@google/genai";
import { Note, ClusterNode, SearchResult } from "../types";
import { SemanticCentroid, HardSample } from "../types";

// ⚙️ CONFIGURATION
const CONFIG = {
  // Model selection: "gemini-2.5-flash-lite" | "gemini-2.5-flash" | "gemini-2.5-pro" | "gemini-3-pro"
  modelName: "gemini-2.5-flash-lite" as const,
};

const getAIClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// Helper to extract text from GenAI response
const extractResponseText = (response: any): string => {
  if (!response) {
    console.warn("❌ Response is null/undefined");
    return "";
  }

  // Log response structure for debugging
  console.log("📦 Response type:", typeof response);

  // Try direct text property first
  if (typeof response.text === "string") {
    console.log(
      "✅ Found text as string property. response.text",
      response.text
    );
    return response.text;
  }

  // Try text() method
  if (typeof response.text === "function") {
    console.log("✅ Found text as function, calling it...");
    const result = response.text();
    console.log(
      "✅ text() returned:",
      typeof result,
      result?.substring?.(0, 100)
    );
    return result || "";
  }

  // Try candidates array (standard Gemini API response)
  if (response.candidates?.[0]?.content?.parts?.[0]?.text) {
    console.log("✅ Found text in candidates array");
    return response.candidates[0].content.parts[0].text;
  }

  console.warn("❌ Could not extract text from response:", response);
  console.log(
    "📦 Response content:",
    JSON.stringify(response).substring(0, 500)
  );
  return "";
};

/**
 * Generate semantic cluster names with enhanced context
 * Used during Phase 3 semantic enhancement
 */
export const generateSemanticClusterName = async (
  notes: Note[],
  centroid?: SemanticCentroid
): Promise<{ name: string; description: string }> => {
  if (notes.length === 0) {
    return { name: "Empty Cluster", description: "No notes" };
  }

  const ai = getAIClient();
  const notesPreview = notes.map((n) => ({
    title: n.title,
    contentSnippet: n.content.substring(0, 150),
  }));

  let semanticContext = "";
  if (centroid) {
    const keywords = centroid.keywords?.join(", ") || "";
    semanticContext = `
Semantic analysis found these themes:
- Description: ${centroid.description || ""}
- Keywords: ${keywords}`;
  }

  const prompt = `
You are a knowledge organization expert.

Analyze these ${notes.length} notes and provide a meaningful cluster name.
${semanticContext}

RULES:
1. The name MUST be a clear, descriptive topic (2-4 words)
2. Focus on the core subject matter that unifies these notes
3. NEVER use generic names like "Miscellaneous", "Topic Group", "Cluster", "Other", or "Uncategorized"
4. If notes are diverse, find the broadest common theme

Example good names: "Machine Learning", "Web Development", "Cooking Recipes", "Personal Finance"
Example bad names: "Cluster 1", "Misc Notes", "Topic Group", "Various Topics"

Notes:
${JSON.stringify(notesPreview, null, 2)}

Return JSON with "name" (2-4 word topic) and "description" (1 sentence).`;

  try {
    const response = await ai.models.generateContent({
      model: CONFIG.modelName,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name: { type: Type.STRING },
            description: { type: Type.STRING },
          },
          required: ["name", "description"],
        },
      },
    });

    const text = extractResponseText(response);
    if (!text) {
      console.warn("No text from semantic cluster naming LLM, using fallback");
      return {
        name: centroid?.description || `Topic Group (${notes.length} notes)`,
        description:
          centroid?.description ||
          notes
            .map((n) => n.title)
            .slice(0, 2)
            .join(", "),
      };
    }

    const parsed = JSON.parse(text);
    return {
      name: parsed.name || centroid?.description || `Topic Group`,
      description:
        parsed.description || centroid?.description || "Related notes",
    };
  } catch (e) {
    console.error("Error generating semantic cluster name:", e);
    return {
      name: centroid?.description || `Topic Group (${notes.length} notes)`,
      description:
        centroid?.description ||
        notes
          .map((n) => n.title)
          .slice(0, 3)
          .join(", "),
    };
  }
};

// --- Generate Cluster Name for a Specific Group (for Embedding Partitions) ---

/**
 * Generate a meaningful name and description for a specific group of notes
 * Used by embedding-based partitioning to name each partition
 */
export const generateClusterName = async (
  notes: Note[]
): Promise<{ name: string; description: string }> => {
  if (notes.length === 0) {
    return { name: "Empty Cluster", description: "No notes" };
  }

  const ai = getAIClient();
  const notesPreview = notes.map((n) => ({
    title: n.title,
    contentSnippet: n.content.substring(0, 150),
  }));

  const prompt = `
You are a knowledge organization expert.

Analyze these ${notes.length} notes and provide a meaningful cluster name.

RULES:
1. The name should be a clear, descriptive topic (2-4 words)
2. Focus on the core subject matter that unifies these notes
3. NEVER use generic names like "Miscellaneous", "Topic Group", "Cluster", or "Other"

Example good names: "Machine Learning", "React Development", "Italian Cooking", "Project Planning"
Example bad names: "Cluster 1", "Misc Notes", "Topic Group", "Various Topics"

Notes:
${JSON.stringify(notesPreview, null, 2)}

Return JSON with "name" (2-4 word topic) and "description" (1 sentence explaining the theme).`;

  try {
    const response = await ai.models.generateContent({
      model: CONFIG.modelName,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name: { type: Type.STRING },
            description: { type: Type.STRING },
          },
          required: ["name", "description"],
        },
      },
    });

    const text = extractResponseText(response);
    if (!text) {
      console.warn("No text from cluster naming LLM, using fallback");
      return {
        name: `Topic Group (${notes.length} notes)`,
        description: notes.map((n) => n.title).join(", "),
      };
    }

    const parsed = JSON.parse(text);
    return {
      name: parsed.name || `Topic Group`,
      description: parsed.description || "Related notes",
    };
  } catch (e) {
    console.error("Error generating cluster name:", e);
    return {
      name: `Topic Group (${notes.length} notes)`,
      description: notes
        .map((n) => n.title)
        .slice(0, 3)
        .join(", "),
    };
  }
};

// --- Dual-Prompt Clustering Logic (Phase 1.2) ---

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
You are an expert knowledge organizer specializing in semantic categorization.

Analyze these notes and group them into meaningful thematic clusters. 

IMPORTANT RULES:
1. Create 2-5 clusters maximum (prefer fewer, broader categories)
2. Each cluster name should be a clear, descriptive topic (2-4 words)
3. Group notes by their core subject matter, not surface-level keywords
4. Every note MUST be assigned to exactly one cluster
5. Prefer semantic similarity over exact keyword matching

Example good cluster names: "Machine Learning", "Web Development", "Cooking Recipes", "Project Management"
Example bad cluster names: "Cluster 1", "Topic Group"

Notes to cluster:
${JSON.stringify(notesLite, null, 2)}

Return a JSON array where each cluster has: id, name, description, noteIds (array of note IDs in this cluster)
  `;

  console.log("🚀 Calling Gemini API for initial clustering...");
  const initialResponse = await ai.models.generateContent({
    model: CONFIG.modelName,
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
    const responseText = extractResponseText(initialResponse);
    initialClusters = JSON.parse(responseText || "[]");
    //print for debugging
    console.log("Initial Clusters:", initialClusters);
  } catch (e) {
    console.error(
      "Initial clustering parse error",
      e,
      "Response:",
      initialResponse
    );
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

  console.log("🚀 Calling Gemini API for refinement clustering...");
  const refinementResponse = await ai.models.generateContent({
    model: CONFIG.modelName,
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
    const responseText = extractResponseText(refinementResponse);
    clusters = JSON.parse(responseText || "[]");
    //print for debugging
    console.log("Refined Clusters:", clusters);
  } catch (e) {
    console.error(
      "Refinement clustering parse error",
      e,
      "Response:",
      refinementResponse
    );
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
    confidence: 0.9,
    method: "refinement",
  };
  //print for debugging
  console.log("Clustering Decision:", decision);

  // 5. Persist decision in memory buffer
  memory.addDecision(decision);

  return { clusters, decision };
};

// --- Auto-Clustering Logic ---

export const clusterNotesWithGemini = async (
  notes: Note[]
): Promise<ClusterNode[]> => {
  const ai = getAIClient();

  const notesLite = notes.map((n) => ({
    id: n.id,
    title: n.title,
    contentSnippet: n.content.substring(0, 200),
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

  const response = await ai.models.generateContent({
    model: CONFIG.modelName,
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

  const jsonStr = extractResponseText(response) || "[]";
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
    model: CONFIG.modelName,
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

  const jsonStr = extractResponseText(response) || "[]";
  try {
    const rawResults = JSON.parse(jsonStr) as {
      noteId: string;
      score: number;
      reason: string;
    }[];

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
    model: CONFIG.modelName,
    contents: prompt,
  });

  return extractResponseText(response)?.trim() || text;
};

// --- Semantic Centroid Generation (Phase 3, Step 3.1) ---

export const generateSemanticCentroid = async (
  clusterId: string,
  clusterNotes: Note[],
  description?: string
): Promise<SemanticCentroid> => {
  const ai = getAIClient();

  const notesSummary = clusterNotes
    .slice(0, 5)
    .map((n) => `- ${n.title}: ${n.content.substring(0, 100)}...`)
    .join("\n");

  //print for debugging
  console.log(
    `Generating semantic centroid for cluster ${clusterId} with ${clusterNotes.length} notes.`
  );
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
      model: CONFIG.modelName,
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

    const result = JSON.parse(extractResponseText(response) || "{}") as {
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
    //print for debugging
    console.log("Centroid details:", centroid);
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

// --- Hard Sample Detection (Phase 3, Step 3.2) ---

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
      model: CONFIG.modelName,
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

    const result = JSON.parse(extractResponseText(response) || "{}");

    if (result.ambiguityScore > 0.6) {
      const hardSample: HardSample = {
        noteId: note.id,
        originalContent: note.content,
        ambiguityScore: Math.max(0, Math.min(1, result.ambiguityScore || 0.5)),
        possibleClusters: result.possibleClusters || clusterIds,
        augmentedContent: result.augmentedContent,
        rewriteReason: result.shouldAugment ? "High ambiguity" : undefined,
      };

      //print for debugging
      console.log("Hard sample details:", hardSample);
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
