import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { ClusterNode, Note } from '../types';

interface ClusterGraphProps {
  clusters: ClusterNode[];
  onNoteSelect: (noteId: string) => void;
}

// Convert hierarchical cluster data to flat nodes/links for D3 (recursive for all levels)
const processData = (clusters: ClusterNode[]) => {
  const nodes: any[] = [];
  const links: any[] = [];
  const seenIds = new Set<string>(); // Track IDs to avoid duplicates

  // Debug: Log input structure
  console.log('📊 ClusterGraph input:', JSON.stringify(clusters, null, 2).substring(0, 2000));

  // Root node
  const rootId = 'graph-root';
  nodes.push({ id: rootId, name: 'Knowledge Base', type: 'root', level: 0, r: 35 });
  seenIds.add(rootId);

  // Recursively process clusters at any depth
  const processCluster = (cluster: ClusterNode, parentId: string, level: number) => {
    // Create unique ID for this cluster - include parent ID to avoid collisions across domains
    const clusterId = `cluster-${parentId}-L${level}-${cluster.id}`;
    
    console.log(`  Processing cluster: "${cluster.name}" (id: ${cluster.id}, type: ${cluster.type}, level: ${level}, children: ${cluster.children?.length || 0})`);
    
    // Skip if already processed (avoid duplicate nodes)
    if (seenIds.has(clusterId)) {
      console.warn(`Duplicate cluster ID: ${clusterId}, skipping`);
      return;
    }
    seenIds.add(clusterId);

    // Determine node type and size based on level
    const isSubcluster = level > 1;
    const nodeType = isSubcluster ? 'subcluster' : 'cluster';
    const radius = Math.max(12, 25 - level * 4);

    nodes.push({ 
      id: clusterId, 
      name: cluster.name, 
      type: nodeType, 
      level,
      r: radius 
    });
    links.push({ source: parentId, target: clusterId, level });

    // Process children (can be notes or nested clusters)
    if (cluster.children) {
      console.log(`    Children of "${cluster.name}":`, cluster.children.map(c => `${c.type}:${c.name}`).join(', '));
      cluster.children.forEach((child, idx) => {
        console.log(`      [${idx}] Processing child: type="${child.type}", name="${child.name}", hasChildren=${!!child.children}, childrenCount=${child.children?.length || 0}`);
        if (child.type === 'note') {
          // Create unique ID for note - include parent cluster ID to avoid collisions
          const noteId = `note-${clusterId}-${child.noteId || child.id}`;
          
          // Skip duplicate notes
          if (seenIds.has(noteId)) {
            return;
          }
          seenIds.add(noteId);

          nodes.push({ 
            id: noteId, 
            name: child.name, 
            type: 'note', 
            noteId: child.noteId,
            level: level + 1,
            r: 8 
          });
          // Link note to its DIRECT parent (the current cluster)
          links.push({ source: clusterId, target: noteId, level: level + 1 });
        } else if (child.type === 'cluster') {
          // Nested cluster - recurse with current cluster as parent
          console.log(`      ➡️ Recursing into cluster: "${child.name}" with ${child.children?.length || 0} children`);
          processCluster(child, clusterId, level + 1);
        } else {
          console.warn(`      ⚠️ Unknown child type: "${child.type}" for "${child.name}"`);
        }
      });
    }
  };

  // Process all top-level clusters
  clusters.forEach(cluster => {
    processCluster(cluster, rootId, 1);
  });

  console.log(`📊 Graph data: ${nodes.length} nodes, ${links.length} links`);
  console.log('📊 All nodes:', nodes.map(n => `${n.type}:${n.name}`).join(', '));
  return { nodes, links };
};

const ClusterGraphComponent: React.FC<ClusterGraphProps> = ({ clusters, onNoteSelect }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!clusters.length || !svgRef.current || !containerRef.current) return;

    const { nodes, links } = processData(clusters);
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous

    // Create a container group for zoom/pan - this is the key fix!
    const container = svg.append("g").attr("class", "zoom-container");

    // Initialize node positions to prevent NaN/undefined issues
    nodes.forEach((node, i) => {
      if (node.type === 'root') {
        node.x = width / 2;
        node.y = height / 2;
      } else {
        // Spread nodes in a circle initially
        const angle = (i / nodes.length) * 2 * Math.PI;
        const radius = 100 + node.level * 50;
        node.x = width / 2 + Math.cos(angle) * radius;
        node.y = height / 2 + Math.sin(angle) * radius;
      }
    });

    // Simulation with improved forces for hierarchical layout
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id((d: any) => d.id)
        .distance((d: any) => 60 + (d.level || 1) * 20) // Longer links for deeper levels
        .strength(0.7))
      .force("charge", d3.forceManyBody()
        .strength((d: any) => d.type === 'root' ? -500 : d.type === 'note' ? -100 : -250))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius((d: any) => d.r + 8).strength(0.8))
      .force("x", d3.forceX(width / 2).strength(0.02))
      .force("y", d3.forceY(height / 2).strength(0.02))
      .alphaDecay(0.02); // Slower decay for better layout

    // Links - drawn in the container group
    const link = container.append("g")
      .attr("class", "links")
      .attr("stroke", "#2e323e")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", (d: any) => Math.max(1, 3 - (d.level || 1)));

    // Node Groups - drawn in the container group
    const node = container.append("g")
      .attr("class", "nodes")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .selectAll("g")
      .data(nodes)
      .join("g")
      .call((d3.drag() as any)
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Node Circles with level-based colors
    node.append("circle")
      .attr("r", (d: any) => d.r)
      .attr("fill", (d: any) => {
        if (d.type === 'root') return '#a855f7';       // Purple
        if (d.type === 'cluster') return '#3b82f6';    // Blue
        if (d.type === 'subcluster') return '#10b981'; // Green for subclusters
        return '#64748b';                               // Gray for notes
      })
      .attr("cursor", "pointer")
      .on("click", (event, d: any) => {
        if (d.type === 'note' && d.noteId) {
          onNoteSelect(d.noteId);
        }
      });

    // Labels
    node.append("text")
      .text((d: any) => d.name.length > 20 ? d.name.slice(0, 18) + '...' : d.name)
      .attr("x", (d: any) => d.r + 5)
      .attr("y", 4)
      .attr("stroke", "none")
      .attr("fill", "#e2e8f0")
      .attr("font-size", (d: any) => {
        if (d.type === 'root') return "14px";
        if (d.type === 'cluster') return "12px";
        if (d.type === 'subcluster') return "11px";
        return "10px";
      })
      .attr("font-weight", (d: any) => d.type === 'note' ? "normal" : "bold")
      .attr("pointer-events", "none");

    simulation.on("tick", () => {
      // Clamp positions to prevent NaN and keep nodes in reasonable bounds
      nodes.forEach(d => {
        if (isNaN(d.x) || d.x === undefined) d.x = width / 2;
        if (isNaN(d.y) || d.y === undefined) d.y = height / 2;
        // Keep nodes within extended bounds
        d.x = Math.max(-width, Math.min(width * 2, d.x));
        d.y = Math.max(-height, Math.min(height * 2, d.y));
      });

      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Zoom - FIXED: only transform the container group, not all g elements
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom as any);

    // Cleanup on unmount
    return () => {
      simulation.stop();
    };

  }, [clusters, onNoteSelect]);

  return (
    <div ref={containerRef} className="w-full h-full bg-background rounded-lg overflow-hidden relative">
      <div className="absolute top-4 left-4 z-10 bg-surface/80 p-2 rounded text-xs text-muted">
        <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-purple-500"></div>Root</div>
        <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-blue-500"></div>Domain</div>
        <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-emerald-500"></div>Subtopic</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-slate-500"></div>Note</div>
      </div>
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

// Memoize to prevent unnecessary re-renders when parent component updates but clusters/onNoteSelect haven't changed
const ClusterGraph = React.memo(ClusterGraphComponent);

export default ClusterGraph;