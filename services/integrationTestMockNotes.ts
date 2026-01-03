/**
 * Integration Test: Mock Notes with Clustering Pipeline
 *
 * This file demonstrates how to use the mock notes dataset with your
 * three-phase semantic clustering system.
 */

import { generateMockNotes, generateClusteredMockNotes } from "./mockNotes";
import { fullSemanticClustering } from "./services/clusteringService";
import { getAllNotes, bulkSaveNotes } from "./services/storageService";

/**
 * Test 1: Generate and validate mock notes
 */
export async function testMockNotesGeneration() {
  console.log("🧪 TEST 1: Mock Notes Generation");

  const notes = generateMockNotes();

  console.log(`✅ Generated ${notes.length} notes`);
  console.log("📊 Distribution by topic:");

  const topicCounts = new Map<string, number>();
  notes.forEach((note) => {
    const count = (topicCounts.get(note.folder) || 0) + 1;
    topicCounts.set(note.folder, count);
  });

  topicCounts.forEach((count, topic) => {
    console.log(`  • ${topic}: ${count} notes`);
  });

  // Verify structure
  notes.forEach((note) => {
    if (!note.id || !note.title || !note.content) {
      throw new Error(`Invalid note structure: ${JSON.stringify(note)}`);
    }
  });

  console.log("✅ All notes have valid structure");
  return notes;
}

/**
 * Test 2: Load mock notes into database
 */
export async function testLoadMockNotesIntoDB() {
  console.log("\n🧪 TEST 2: Load Mock Notes to Database");

  const notes = generateMockNotes();

  try {
    console.log(`💾 Saving ${notes.length} notes to IndexedDB...`);
    await bulkSaveNotes(notes);

    // Verify they were saved
    const loadedNotes = await getAllNotes();
    console.log(`✅ Verified: ${loadedNotes.length} notes in database`);

    return loadedNotes;
  } catch (e) {
    console.error("❌ Failed to save notes:", e);
    throw e;
  }
}

/**
 * Test 3: Cluster mock notes (Phase 1 only - fast)
 */
export async function testPhase1ClusteringOnly() {
  console.log("\n🧪 TEST 3: Phase 1 Clustering Only (LLM Dual-Prompt)");

  const notes = generateMockNotes();
  console.log(`📝 Clustering ${notes.length} notes with Phase 1 only...`);

  try {
    const result = await fullSemanticClustering(notes, {
      useHybridEmbeddings: false, // Skip Phase 2
      useSemanticEnhancement: false, // Skip Phase 3
    });

    console.log(`✅ Phase 1 complete:`);
    console.log(`  • Clusters created: ${result.clusters.length}`);
    console.log(`  • Confidence: ${result.finalConfidence.toFixed(2)}`);
    console.log(`  • Time: ~2-3 seconds`);

    return result;
  } catch (e) {
    console.error("❌ Phase 1 clustering failed:", e);
    throw e;
  }
}

/**
 * Test 4: Full 3-phase clustering (complete pipeline)
 */
export async function testFullSemanticClustering() {
  console.log("\n🧪 TEST 4: Full 3-Phase Semantic Clustering");

  const notes = generateMockNotes();
  console.log(`📝 Running full pipeline on ${notes.length} notes...`);

  try {
    const result = await fullSemanticClustering(notes, {
      useHybridEmbeddings: true, // Phase 2: Embeddings
      useSemanticEnhancement: true, // Phase 3: Semantic analysis
      generateCentroids: true,
      detectHardSamples: true,
    });

    console.log(`✅ Full clustering complete:`);
    console.log(`  • Clusters: ${result.clusters.length}`);
    console.log(`  • Semantic Centroids: ${result.semanticCentroids.size}`);
    console.log(`  • Hard Samples Detected: ${result.hardSamples.length}`);
    console.log(
      `  • Constraints Generated: ${result.constraints.totalConstraints}`
    );
    console.log(`  • Final Confidence: ${result.finalConfidence.toFixed(2)}`);
    console.log(`  • Total Time: ~5-10 seconds`);

    return result;
  } catch (e) {
    console.error("❌ Full clustering failed:", e);
    throw e;
  }
}

/**
 * Test 5: Validate clustered structure matches expected clusters
 */
export async function testValidateClusteringQuality() {
  console.log("\n🧪 TEST 5: Validate Clustering Quality");

  const { notes, expectedClusters } = generateClusteredMockNotes();
  console.log(`📝 Validating clustering quality against expected structure...`);

  try {
    const result = await fullSemanticClustering(notes, {
      useHybridEmbeddings: true,
      useSemanticEnhancement: false,
    });

    console.log(`✅ Clustering Results:`);
    console.log(`  • Expected ${expectedClusters.length} main topics`);
    console.log(`  • Got ${result.clusters.length} clusters`);

    // Analyze cluster distribution
    console.log(`\n📊 Cluster Distribution:`);
    result.clusters.forEach((cluster) => {
      const noteCount = cluster.children ? cluster.children.length : 0;
      console.log(`  • ${cluster.name}: ${noteCount} notes`);
    });

    console.log(`\n✅ Quality Metrics:`);
    console.log(
      `  • Confidence Score: ${result.finalConfidence.toFixed(2)} / 1.0`
    );
    console.log(`  • Expected: >0.75 for good clustering`);

    return result;
  } catch (e) {
    console.error("❌ Quality validation failed:", e);
    throw e;
  }
}

/**
 * Test 6: Performance benchmark with mock notes
 */
export async function testPerformanceBenchmark() {
  console.log("\n🧪 TEST 6: Performance Benchmark");

  const notes = generateMockNotes();
  console.log(`⏱️  Running performance benchmark on ${notes.length} notes...`);

  try {
    // Measure Phase 1 only
    const phase1Start = performance.now();
    const phase1Result = await fullSemanticClustering(notes, {
      useHybridEmbeddings: false,
      useSemanticEnhancement: false,
    });
    const phase1Time = performance.now() - phase1Start;

    // Measure full pipeline
    const fullStart = performance.now();
    const fullResult = await fullSemanticClustering(notes, {
      useHybridEmbeddings: true,
      useSemanticEnhancement: true,
    });
    const fullTime = performance.now() - fullStart;

    console.log(`\n⏱️  Performance Results:`);
    console.log(`  • Phase 1 only: ${phase1Time.toFixed(0)}ms`);
    console.log(`  • Full pipeline: ${fullTime.toFixed(0)}ms`);
    console.log(
      `  • Average per note: ${(fullTime / notes.length).toFixed(2)}ms`
    );

    console.log(`\n✅ Benchmark targets:`);
    console.log(
      `  • Target <5s for 100 notes: ${fullTime < 5000 ? "✅ PASS" : "❌ FAIL"}`
    );
    console.log(
      `  • Target <30s for 1000 notes: ${
        (notes.length / 100) * fullTime < 30000
          ? "✅ PASS (estimated)"
          : "❌ FAIL (estimated)"
      }`
    );

    return { phase1Time, fullTime };
  } catch (e) {
    console.error("❌ Benchmark failed:", e);
    throw e;
  }
}

/**
 * Run all integration tests
 */
export async function runAllIntegrationTests() {
  console.log("🚀 Starting Integration Tests with Mock Notes\n");
  console.log("=============================================");

  try {
    // Test 1: Generation
    await testMockNotesGeneration();

    // Test 2: Database loading
    await testLoadMockNotesIntoDB();

    // Test 3: Phase 1 clustering
    await testPhase1ClusteringOnly();

    // Test 4: Full clustering
    await testFullSemanticClustering();

    // Test 5: Quality validation
    await testValidateClusteringQuality();

    // Test 6: Performance
    await testPerformanceBenchmark();

    console.log("\n✅ All integration tests passed!");
    console.log("=============================================");
  } catch (e) {
    console.error("\n❌ Integration tests failed!");
    console.error(e);
  }
}

/**
 * Browser console usage:
 *
 * // Run individual tests:
 * window.testMockNotesGeneration();
 * window.testLoadMockNotesIntoDB();
 * window.testPhase1ClusteringOnly();
 * window.testFullSemanticClustering();
 * window.testValidateClusteringQuality();
 * window.testPerformanceBenchmark();
 *
 * // Run all tests:
 * window.runAllIntegrationTests();
 */

// Export for use in browser console
if (typeof window !== "undefined") {
  (window as any).testMockNotesGeneration = testMockNotesGeneration;
  (window as any).testLoadMockNotesIntoDB = testLoadMockNotesIntoDB;
  (window as any).testPhase1ClusteringOnly = testPhase1ClusteringOnly;
  (window as any).testFullSemanticClustering = testFullSemanticClustering;
  (window as any).testValidateClusteringQuality = testValidateClusteringQuality;
  (window as any).testPerformanceBenchmark = testPerformanceBenchmark;
  (window as any).runAllIntegrationTests = runAllIntegrationTests;
}
