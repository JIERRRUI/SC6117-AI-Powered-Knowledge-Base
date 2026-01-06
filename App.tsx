import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Note, SearchMode, SearchResult, ProcessingStatus, ClusterNode } from './types';
import { clusterNotesWithGemini, correctTextWithGemini } from './services/geminiService';
import { executeExactSearch, executeHybridSearch } from './services/searchService';
import { getAllNotes, saveNote, deleteNote, bulkSaveNotes } from './services/storageService';
import {
  SearchIcon,
  FileTextIcon,
  NetworkIcon,
  ZapIcon,
  LayersIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  FolderIcon,
  WifiOffIcon,
  UploadCloudIcon,
  XIcon,
  PlusIcon,
  WandIcon,
  TrashIcon,
  SaveIcon,
  ExportPdfIcon
} from './components/Icons';
import ClusterGraph from './components/ClusterGraph';
import ConfirmationModal from './components/ConfirmationModal';

import * as pdfjsLib from 'pdfjs-dist';
import mammoth from 'mammoth';
import { jsPDF } from 'jspdf';

// -----------------------------------------------------------------------------
// PDF.js worker config (Vite-friendly)
//
// Using the bundled worker via `new URL(..., import.meta.url)` is the most robust
// approach for Vite. Avoids CDN/CORS/version mismatch.
// -----------------------------------------------------------------------------
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

// --- Helper Functions for File Processing ---

// Convert HTML (like OneNote exports) to Markdown
const convertHtmlToMarkdown = (html: string): string => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');

  const processNode = (node: Node): string => {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.textContent?.replace(/\s+/g, ' ') || '';
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return '';

    const el = node as HTMLElement;
    const childrenContent = Array.from(el.childNodes).map(processNode).join('');

    switch (el.tagName.toLowerCase()) {
      case 'h1':
        return `\n# ${childrenContent}\n\n`;
      case 'h2':
        return `\n## ${childrenContent}\n\n`;
      case 'h3':
        return `\n### ${childrenContent}\n\n`;
      case 'p':
        return `${childrenContent}\n\n`;
      case 'div':
        return `${childrenContent}\n`;
      case 'ul':
      case 'ol':
        return `${childrenContent}\n`;
      case 'li':
        return `- ${childrenContent.trim()}\n`;
      case 'br':
        return '\n';
      case 'b':
      case 'strong':
        return ` **${childrenContent.trim()}** `;
      case 'i':
      case 'em':
        return ` *${childrenContent.trim()}* `;
      case 'a':
        return ` [${childrenContent.trim()}](${(el as HTMLAnchorElement).href}) `;
      case 'code':
      case 'pre':
        return ` \`${childrenContent}\` `;
      case 'head':
      case 'style':
      case 'script':
        return '';
      case 'body':
        return childrenContent;
      default:
        return childrenContent;
    }
  };

  const rawMarkdown = processNode(doc.body);
  return rawMarkdown.replace(/\n\s+\n/g, '\n\n').trim();
};

// Extract text from PDF
const extractTextFromPdf = async (file: File): Promise<string> => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    let fullText = '';
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = (textContent.items as any[])
        .map((item: any) => item.str)
        .join(' ');
      fullText += `## Page ${i}\n\n${pageText}\n\n`;
    }

    return fullText;
  } catch (err) {
    console.error('PDF Extraction error:', err);
    return 'Failed to extract text from PDF.';
  }
};

// Extract Markdown from Word (DOCX)
const extractTextFromDocx = async (file: File): Promise<string> => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.convertToMarkdown({ arrayBuffer });
    return result.value;
  } catch (err) {
    console.error('DOCX Extraction error:', err);
    return 'Failed to extract text from Word document.';
  }
};

// Extract readable text from binary files (like .one)
const extractTextFromBinary = async (file: File): Promise<string> => {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let result = '';
  let currentRun = '';

  for (let i = 0; i < bytes.length; i++) {
    const code = bytes[i];
    if ((code >= 32 && code <= 126) || code === 10 || code === 13 || code === 9) {
      currentRun += String.fromCharCode(code);
    } else {
      if (currentRun.length >= 4) result += currentRun + '\n';
      currentRun = '';
    }
  }
  if (currentRun.length >= 4) result += currentRun;

  return result.length > 0
    ? result
    : 'Binary content detected but no readable text could be extracted.';
};

// Very small Markdown -> plain text helper for PDF export.
// (Keeps content readable; not a full Markdown renderer.)
const markdownToPlainText = (md: string): string => {
  return md
    // code blocks
    .replace(/```[\s\S]*?```/g, match => {
      const inner = match.replace(/```/g, '');
      return `\n${inner}\n`;
    })
    // inline code
    .replace(/`([^`]+)`/g, '$1')
    // images: ![alt](url) -> alt (url)
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '$1 ($2)')
    // links: [text](url) -> text (url)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1 ($2)')
    // headings
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    // blockquotes
    .replace(/^\s{0,3}>\s?/gm, '')
    // list markers
    .replace(/^\s*[-*+]\s+/gm, '• ')
    .replace(/^\s*\d+\.\s+/gm, '• ')
    // emphasis
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    // horizontal rules
    .replace(/^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$/gm, '')
    // normalize whitespace
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

// --- Helper Component: Search Result Item ---

interface SearchResultItemProps {
  result: SearchResult;
  onClick: () => void;
}

const SearchResultItem: React.FC<SearchResultItemProps> = ({ result, onClick }) => (
  <div
    onClick={onClick}
    className="p-3 mb-2 rounded bg-surface border border-border hover:border-primary cursor-pointer transition-colors group"
  >
    <div className="flex justify-between items-start">
      <h4 className="font-semibold text-text group-hover:text-primary transition-colors">
        {result.note.title}
      </h4>
      <span className="text-xs font-mono px-2 py-0.5 rounded bg-black/30 text-muted">
        {Math.round(result.score)}
        {result.reason ? '/100' : ''}
      </span>
    </div>
    {result.highlight ? (
      <div
        className="text-sm text-muted mt-1 line-clamp-2"
        dangerouslySetInnerHTML={{ __html: result.highlight }}
      />
    ) : (
      <div className="text-sm text-muted mt-1 line-clamp-2">
        {result.note.content.substring(0, 150)}...
      </div>
    )}
    {result.reason && (
      <div className="mt-2 text-xs text-secondary bg-secondary/10 p-2 rounded border border-secondary/20">
        <span className="font-bold">AI Reason:</span> {result.reason}
      </div>
    )}
    <div className="mt-2 flex gap-2">
      {result.note.tags.map(tag => (
        <span
          key={tag}
          className="text-[10px] uppercase font-bold tracking-wider text-muted bg-border/50 px-1.5 py-0.5 rounded"
        >
          {tag}
        </span>
      ))}
    </div>
  </div>
);

// --- Helper Component: File Tree Node ---

interface FileTreeNodeProps {
  node: ClusterNode;
  depth?: number;
  activeId?: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({
  node,
  depth = 0,
  activeId,
  onSelect,
  onDelete
}) => {
  const [expanded, setExpanded] = useState(true);

  if (node.type === 'note') {
    return (
      <div
        onClick={() => node.noteId && onSelect(node.noteId)}
        className={`group flex items-center gap-2 py-1 px-2 cursor-pointer text-sm transition-colors ${
          activeId === node.noteId
            ? 'bg-primary/20 text-primary'
            : 'text-muted hover:text-text hover:bg-white/5'
        }`}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
      >
        <FileTextIcon className="w-4 h-4 opacity-70" />
        <span className="truncate flex-1">{node.name}</span>
        {node.noteId && (
          <button
            onClick={e => {
              e.stopPropagation();
              e.preventDefault();
              onDelete(node.noteId!);
            }}
            className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 hover:text-red-400 text-muted transition-all rounded"
            title="Delete note"
          >
            <TrashIcon className="w-3 h-3" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div>
      <div
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 py-1 px-2 cursor-pointer text-sm text-text hover:bg-white/5 transition-colors font-medium select-none"
        style={{ paddingLeft: `${depth * 16}px` }}
      >
        {expanded ? (
          <ChevronDownIcon className="w-4 h-4 text-muted" />
        ) : (
          <ChevronRightIcon className="w-4 h-4 text-muted" />
        )}
        <FolderIcon className="w-4 h-4 text-primary" />
        <span className="truncate">{node.name}</span>
      </div>
      {expanded && node.children && (
        <div>
          {node.children.map(child => (
            <FileTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              activeId={activeId}
              onSelect={onSelect}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// --- Helper Component: Upload Modal ---

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (files: FileList) => void;
}

const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) onUpload(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-xl shadow-2xl w-full max-w-lg overflow-hidden relative animate-in fade-in zoom-in duration-200">
        <button onClick={onClose} className="absolute top-4 right-4 text-muted hover:text-white transition-colors">
          <XIcon className="w-5 h-5" />
        </button>

        <div className="p-8 text-center">
          <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-6 text-primary">
            <UploadCloudIcon className="w-8 h-8" />
          </div>
          <h2 className="text-2xl font-bold text-text mb-2">Import Notes</h2>
          <p className="text-muted text-sm mb-8">
            Upload notes to add them to your knowledge base.
            <br />
            Supports <b>Markdown, Text, PDF, Word (.docx), OneNote (.one), and HTML</b>.
          </p>

          <div
            className={`border-2 border-dashed rounded-lg p-10 transition-colors cursor-pointer ${
              isDragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-white/5'
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              multiple
              // FIX: removed typos/duplicates (e.g., "..md")
              accept=".md,.txt,.markdown,.html,.htm,.one,.pdf,.docx"
              onChange={e => {
                if (e.target.files) onUpload(e.target.files);
              }}
            />
            <p className="text-sm font-medium text-text">Click to browse or drag files here</p>
            <p className="text-xs text-muted mt-2">.md, .txt, .html, .one, .pdf, .docx</p>
          </div>
        </div>
        <div className="bg-black/20 p-4 text-center text-xs text-muted border-t border-border">
          Processed locally in your browser
        </div>
      </div>
    </div>
  );
};

const App = () => {
  // State
  const [notes, setNotes] = useState<Note[]>([]);
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>(SearchMode.EXACT);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<ProcessingStatus>({ isProcessing: false, message: '' });
  const [viewMode, setViewMode] = useState<'editor' | 'graph'>('editor');
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isDbLoaded, setIsDbLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // IMPORTANT FIX: keep latest notes available for async saves
  const notesRef = useRef<Note[]>([]);
  useEffect(() => {
    notesRef.current = notes;
  }, [notes]);

  // Clustering State
  const [clusters, setClusters] = useState<ClusterNode[]>([]);
  const [hasClustered, setHasClustered] = useState(false);

  // Folders State for Sidebar
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());

  // Derived State
  const activeNote = useMemo(() => notes.find(n => n.id === activeNoteId), [activeNoteId, notes]);

  // Group notes by folder
  const notesByFolder = useMemo(() => {
    const groups: { [key: string]: Note[] } = {};
    notes.forEach(note => {
      const folder = note.folder || '/misc';
      if (!groups[folder]) groups[folder] = [];
      groups[folder].push(note);
    });
    return Object.entries(groups).sort((a, b) => a[0].localeCompare(b[0]));
  }, [notes]);

  const [moveRequest, setMoveRequest] = useState<{
    noteId: string;
    noteTitle: string;
    targetClusterId: string;
    targetClusterName: string;
  } | null>(null);

  const handleNodeReparent = (noteId: string, newClusterId: string, newClusterName: string) => {
    const targetNote = notes.find(n => n.id === noteId);
    if (!targetNote) return;

    setMoveRequest({
      noteId,
      noteTitle: targetNote.title,
      targetClusterId: newClusterId,
      targetClusterName: newClusterName
    });
  };

  const executeMove = async () => {
    if (!moveRequest) return;
    const { noteId, targetClusterName, targetClusterId } = moveRequest;

    const newFolderPath = `/${targetClusterName.replace(/[\\/:*?"<>|]/g, '')}`;

    await handleUpdateNote(noteId, 'folder', newFolderPath);

    setClusters(prev => {
      return prev.map(cluster => {
        if (cluster.children?.some(c => c.noteId === noteId) && cluster.id !== targetClusterId) {
          return {
            ...cluster,
            children: cluster.children.filter(c => c.noteId !== noteId)
          };
        }
        if (cluster.id === targetClusterId) {
          const noteNode = {
            id: `node-${noteId}`,
            name: moveRequest.noteTitle,
            type: 'note' as const,
            noteId
          };
          return {
            ...cluster,
            children: [...(cluster.children || []), noteNode]
          };
        }
        return cluster;
      });
    });

    setMoveRequest(null);
  };

  // Load notes from DB on startup
  useEffect(() => {
    const loadNotes = async () => {
      try {
        const storedNotes = await getAllNotes();
        setNotes(storedNotes);
        setIsDbLoaded(true);
        const allFolders = new Set(storedNotes.map(n => n.folder || '/misc'));
        setExpandedFolders(allFolders);
      } catch (e) {
        console.error('Failed to load notes', e);
        setStatus({ isProcessing: false, message: 'Error loading database' });
      }
    };
    loadNotes();
  }, []);

  // Offline detection
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => {
      setIsOnline(false);
      if (searchMode !== SearchMode.EXACT) setSearchMode(SearchMode.EXACT);
    };
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [searchMode]);

  const handleSearch = useCallback(async () => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setSearchResults([]);
      return;
    }

    const effectiveMode = !isOnline ? SearchMode.EXACT : searchMode;

    if (!isOnline && searchMode !== SearchMode.EXACT) setSearchMode(SearchMode.EXACT);

    setStatus({
      isProcessing: true,
      message: effectiveMode === SearchMode.EXACT ? 'Searching...' : 'Searching with hybrid AI...'
    });

    try {
      const results: SearchResult[] =
        effectiveMode === SearchMode.EXACT
          ? executeExactSearch(trimmed, notes, { mode: SearchMode.EXACT, useRegex: true })
          : await executeHybridSearch(trimmed, notes, { mode: effectiveMode, useRegex: true });

      setSearchResults(results);
    } catch (e) {
      console.error(e);
      alert('Search failed. Check console.');
    } finally {
      setStatus({ isProcessing: false, message: '' });
    }
  }, [searchQuery, searchMode, notes, isOnline]);

  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(handleSearch, 300);
    return () => clearTimeout(timer);
  }, [searchQuery, searchMode, handleSearch]);

  const triggerSemanticSearch = () => {
    handleSearch();
  };

  const handleCluster = async () => {
    if (!isOnline) return;
    setStatus({ isProcessing: true, message: 'AI is organizing your knowledge base...' });
    try {
      const clusterData = await clusterNotesWithGemini(notes);
      setClusters(clusterData);
      setHasClustered(true);
      setViewMode('graph');
    } catch (e) {
      console.error(e);
      alert('Clustering failed');
    } finally {
      setStatus({ isProcessing: false, message: '' });
    }
  };

  const processFiles = async (files: FileList) => {
    setIsUploadModalOpen(false);
    setStatus({ isProcessing: true, message: 'Importing notes...' });

    const newNotes: Note[] = [];
    const dateStr = new Date().toISOString().split('T')[0];

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        let content = '';
        const name = file.name;

        let title = name.replace(/\.(md|txt|html|htm|one|markdown|pdf|docx)$/i, '');

        if (name.match(/\.(md|txt|markdown)$/i)) {
          const text = await file.text();
          content = text;
          const firstLine = text.split('\n')[0].trim();
          if (firstLine.startsWith('# ')) title = firstLine.substring(2);
        } else if (name.match(/\.(html|htm)$/i)) {
          const rawHtml = await file.text();
          content = convertHtmlToMarkdown(rawHtml);
          if (!content.startsWith('#')) content = `# ${title}\n\n${content}`;
        } else if (name.match(/\.one$/i)) {
          const extractedText = await extractTextFromBinary(file);
          content = `# ${title} (OneNote Extract)\n\n> **Note:** This content was extracted from a binary OneNote file. Formatting may be lost.\n\n${extractedText}`;
        } else if (name.match(/\.pdf$/i)) {
          const extracted = await extractTextFromPdf(file);
          content = `# ${title} (PDF)\n\n${extracted}`;
        } else if (name.match(/\.docx$/i)) {
          const extractedMd = await extractTextFromDocx(file);
          content = extractedMd.trim().startsWith('#') ? extractedMd : `# ${title}\n\n${extractedMd}`;
        }

        if (content) {
          newNotes.push({
            id: `imported-${Date.now()}-${i}`,
            title,
            content,
            tags: ['imported'],
            createdAt: dateStr,
            folder: '/uploads'
          });
        }
      }

      if (newNotes.length > 0) {
        await bulkSaveNotes(newNotes);
        setNotes(prev => [...prev, ...newNotes]);
        setExpandedFolders(prev => new Set(prev).add('/uploads'));

        if (hasClustered) {
          setHasClustered(false);
          setClusters([]);
        }
      }
    } catch (error) {
      console.error('Error reading files', error);
      alert('Failed to import some files.');
    } finally {
      setStatus({ isProcessing: false, message: '' });
    }
  };

  const handleCreateNote = async () => {
    const newNote: Note = {
      id: `new-${Date.now()}`,
      title: 'Untitled Note',
      content: '# New Note\n\nStart typing here...',
      tags: [],
      createdAt: new Date().toISOString().split('T')[0],
      folder: '/drafts'
    };

    setNotes(prev => [newNote, ...prev]);
    setActiveNoteId(newNote.id);
    setViewMode('editor');
    setExpandedFolders(prev => new Set(prev).add('/drafts'));

    if (hasClustered) {
      setHasClustered(false);
      setClusters([]);
    }

    try {
      await saveNote(newNote);
    } catch (e) {
      console.error('Failed to save new note', e);
      alert('Could not save note to database');
    }
  };

  // FIX: uses notesRef to avoid stale saves
  const handleUpdateNote = async (id: string, field: 'title' | 'content' | 'folder', value: string) => {
    setNotes(prev => prev.map(note => (note.id === id ? { ...note, [field]: value } : note)));

    const current = notesRef.current.find(n => n.id === id);
    if (!current) return;

    const updatedNote = { ...current, [field]: value };
    try {
      await saveNote(updatedNote);
    } catch (e) {
      console.error('Background save failed', e);
    }
  };

  const handleManualSave = async () => {
    if (!activeNote) return;
    setIsSaving(true);
    try {
      await saveNote(activeNote);
      setTimeout(() => setIsSaving(false), 1000);
    } catch (e) {
      console.error('Manual save failed', e);
      setIsSaving(false);
      alert('Failed to save');
    }
  };

  const handleExportPdf = () => {
    if (!activeNote) return;

    try {
      const doc = new jsPDF({ unit: 'pt', format: 'a4' });

      const margin = 48;
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const maxWidth = pageWidth - margin * 2;

      const safeTitle = (activeNote.title || 'note')
        .replace(/[\\/:*?"<>|]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      // Header
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(18);
      const titleLines = doc.splitTextToSize(safeTitle || 'Untitled Note', maxWidth);
      let y = margin;
      doc.text(titleLines, margin, y);

      y += titleLines.length * 22;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      const meta = `${activeNote.folder || ''}  •  ${activeNote.createdAt || ''}`;
      doc.text(meta, margin, y);
      y += 18;

      // Body
      doc.setFontSize(11);
      const bodyText = markdownToPlainText(activeNote.content || '');
      const bodyLines = doc.splitTextToSize(bodyText, maxWidth);

      const lineHeight = 16;
      for (let i = 0; i < bodyLines.length; i++) {
        if (y + lineHeight > pageHeight - margin) {
          doc.addPage();
          y = margin;
        }
        doc.text(bodyLines[i], margin, y);
        y += lineHeight;
      }

      doc.save(`${safeTitle || 'note'}.pdf`);
    } catch (e) {
      console.error('Export PDF failed', e);
      alert('Failed to export PDF. Check console.');
    }
  };

  const handleDeleteNote = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this note?')) {
      setNotes(prev => prev.filter(n => n.id !== id));
      if (activeNoteId === id) setActiveNoteId(null);
      if (hasClustered) {
        setHasClustered(false);
        setClusters([]);
      }

      try {
        await deleteNote(id);
      } catch (e) {
        console.error('Delete failed', e);
        alert('Failed to delete from database. Refreshing page might restore it.');
      }
    }
  };

  const handleAICorrect = async () => {
    if (!activeNote || !isOnline) return;
    setStatus({ isProcessing: true, message: 'Fixing grammar and spelling...' });
    try {
      const correctedText = await correctTextWithGemini(activeNote.content);
      await handleUpdateNote(activeNote.id, 'content', correctedText);
    } catch (e) {
      console.error('Correction failed', e);
      alert('AI Correction failed.');
    } finally {
      setStatus({ isProcessing: false, message: '' });
    }
  };

  const toggleFolder = (folder: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folder)) next.delete(folder);
      else next.add(folder);
      return next;
    });
  };

  const renderContent = () => {
    if (status.isProcessing && viewMode === 'graph' && !clusters.length) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-muted">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
          <p>{status.message}</p>
        </div>
      );
    }

    if (viewMode === 'graph' && clusters.length > 0) {
      return (
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-border flex justify-between items-center bg-surface">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <NetworkIcon className="text-primary" /> Knowledge Graph
            </h2>
            <button onClick={() => setViewMode('editor')} className="text-sm text-muted hover:text-text">
              Close Graph
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <ClusterGraph
              clusters={clusters}
              onNoteSelect={id => {
                setActiveNoteId(id);
                setViewMode('editor');
              }}
              onNodeReparent={handleNodeReparent}
            />
          </div>
        </div>
      );
    }

    if (activeNote) {
      return (
        <div className="h-full flex flex-col max-w-3xl mx-auto w-full p-6">
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-sm text-muted">
                <span>{activeNote.folder}</span>
                <span>/</span>
                <span>{activeNote.createdAt}</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleManualSave}
                  disabled={isSaving}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-colors text-xs font-medium border ${
                    isSaving
                      ? 'bg-green-500/10 text-green-500 border-green-500/20'
                      : 'bg-primary/10 text-primary hover:bg-primary/20 border-primary/20'
                  }`}
                  title="Save Note"
                >
                  <SaveIcon className="w-3 h-3" />
                  {isSaving ? 'Saved' : 'Save'}
                </button>

                <button
                  onClick={handleExportPdf}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 text-muted hover:text-white hover:bg-white/10 transition-colors text-xs font-medium border border-border"
                  title="Export this note to PDF"
                >
                  <ExportPdfIcon className="w-3 h-3" />
                  Export PDF
                </button>

                {isOnline && (
                  <button
                    onClick={handleAICorrect}
                    disabled={status.isProcessing}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary/10 text-secondary hover:bg-secondary/20 transition-colors text-xs font-medium border border-secondary/20"
                    title="Fix grammar, spelling and punctuation"
                  >
                    <WandIcon className="w-3 h-3" />
                    {status.isProcessing && status.message.includes('Fixing') ? 'Polishing...' : 'AI Fix Grammar'}
                  </button>
                )}
                <button
                  onClick={() => handleDeleteNote(activeNote.id)}
                  className="p-1.5 rounded-full hover:bg-red-500/10 hover:text-red-400 text-muted transition-colors"
                  title="Delete Note"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            </div>

            <input
              className="w-full bg-transparent text-4xl font-bold text-text mb-4 focus:outline-none placeholder-gray-600"
              value={activeNote.title}
              onChange={e => handleUpdateNote(activeNote.id, 'title', e.target.value)}
              placeholder="Note Title"
            />

            <div className="flex gap-2 mb-6">
              {activeNote.tags.map(tag => (
                <span
                  key={tag}
                  className="text-xs font-medium px-2 py-1 rounded bg-primary/10 text-primary uppercase tracking-wide"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </div>

          <div className="flex-1 h-full">
            <textarea
              className="w-full h-full bg-transparent resize-none focus:outline-none font-mono text-sm leading-relaxed text-slate-300"
              value={activeNote.content}
              onChange={e => handleUpdateNote(activeNote.id, 'content', e.target.value)}
              placeholder="Write your note here..."
            />
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col items-center justify-center h-full text-muted opacity-50">
        <LayersIcon className="w-16 h-16 mb-4" />
        <p>Select a note or search to begin</p>
      </div>
    );
  };

  return (
    <div className="flex h-screen w-full bg-background text-text overflow-hidden">
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUpload={processFiles}
      />

      {/* Sidebar */}
      <div className="w-64 border-r border-border bg-surface flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-bold text-xl tracking-tight text-white mb-1">
              <ZapIcon className="text-yellow-400 fill-current" /> Synapse
            </div>
            {!isOnline && (
              <div className="group relative">
                <WifiOffIcon className="w-4 h-4 text-red-500" />
                <div className="absolute left-full ml-2 top-0 bg-black text-xs p-1 rounded whitespace-nowrap hidden group-hover:block z-50">
                  Offline Mode
                </div>
              </div>
            )}
          </div>
          <p className="text-xs text-muted">Intelligent Knowledge Base</p>
        </div>

        {/* Action Bar */}
        <div className="p-2 border-b border-border grid grid-cols-2 gap-2">
          <button
            onClick={handleCreateNote}
            className="col-span-2 flex items-center justify-center gap-2 p-2 rounded text-xs font-bold transition-all bg-primary hover:bg-blue-600 text-white shadow-lg shadow-blue-900/20"
          >
            <PlusIcon className="w-3 h-3" />
            New Note
          </button>
          <button
            onClick={() => setIsUploadModalOpen(true)}
            className="flex items-center justify-center gap-2 p-2 rounded text-xs font-bold transition-all bg-white/5 hover:bg-white/10 text-muted hover:text-white"
          >
            <UploadCloudIcon className="w-3 h-3" />
            Import
          </button>
          <button
            onClick={handleCluster}
            disabled={status.isProcessing || !isOnline}
            className={`flex items-center justify-center gap-2 p-2 rounded text-xs font-bold transition-all ${
              viewMode === 'graph'
                ? 'bg-secondary text-white'
                : !isOnline
                ? 'bg-white/5 text-muted cursor-not-allowed opacity-50'
                : 'bg-white/5 hover:bg-white/10 text-muted'
            }`}
            title={!isOnline ? 'Unavailable offline' : 'Cluster AI'}
          >
            <NetworkIcon className="w-3 h-3" />
            {status.isProcessing && !hasClustered ? 'Thinking...' : !isOnline ? 'Offline' : 'Cluster AI'}
          </button>
        </div>

        {/* File Explorer */}
        <div className="flex-1 overflow-y-auto p-2">
          <div className="text-xs font-bold text-muted uppercase tracking-wider mb-2 pl-2">
            {clusters.length > 0 ? 'AI Clusters' : 'Files'}
          </div>

          {!isDbLoaded ? (
            <div className="pl-2 text-xs text-muted">Loading database...</div>
          ) : clusters.length > 0 ? (
            <div className="space-y-1">
              {clusters.map(cluster => (
                <FileTreeNode
                  key={cluster.id}
                  node={cluster}
                  activeId={activeNoteId || ''}
                  onSelect={id => {
                    setActiveNoteId(id);
                    setViewMode('editor');
                  }}
                  onDelete={handleDeleteNote}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-1">
              {notesByFolder.map(([folder, folderNotes]) => {
                const isExpanded = expandedFolders.has(folder);
                return (
                  <div key={folder}>
                    <div
                      onClick={() => toggleFolder(folder)}
                      className="flex items-center gap-1 py-1 px-2 cursor-pointer text-sm text-text hover:bg-white/5 transition-colors font-medium select-none"
                    >
                      {isExpanded ? (
                        <ChevronDownIcon className="w-4 h-4 text-muted" />
                      ) : (
                        <ChevronRightIcon className="w-4 h-4 text-muted" />
                      )}
                      <FolderIcon className="w-4 h-4 text-primary" />
                      <span className="truncate">{folder.replace(/^\//, '')}</span>
                    </div>

                    {isExpanded && (
                      <div className="pl-3 border-l border-border ml-3 mt-1 space-y-1">
                        {folderNotes.map(note => (
                          <div
                            key={note.id}
                            onClick={() => {
                              setActiveNoteId(note.id);
                              setViewMode('editor');
                            }}
                            className={`group flex items-center gap-2 p-1.5 rounded cursor-pointer text-sm ${
                              activeNoteId === note.id ? 'bg-primary/20 text-primary' : 'text-muted hover:text-text'
                            }`}
                          >
                            <FileTextIcon className="w-4 h-4 opacity-70" />
                            <span className="truncate flex-1">{note.title}</span>
                            <button
                              onClick={e => {
                                e.stopPropagation();
                                e.preventDefault();
                                handleDeleteNote(note.id);
                              }}
                              className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 hover:text-red-400 text-muted transition-all rounded z-10"
                              title="Delete note"
                            >
                              <TrashIcon className="w-3 h-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Search Bar */}
        <div className="h-16 border-b border-border flex items-center px-6 gap-4 bg-background/50 backdrop-blur-sm z-20 sticky top-0">
          <div className="relative flex-1 max-w-2xl group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <SearchIcon
                className={`w-5 h-5 transition-colors ${
                  status.isProcessing ? 'text-primary animate-pulse' : 'text-muted group-focus-within:text-text'
                }`}
              />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && triggerSemanticSearch()}
              placeholder={searchMode === SearchMode.EXACT ? 'Grep search...' : 'Ask your knowledge base...'}
              className="block w-full pl-10 pr-3 py-2 bg-surface border border-border rounded-lg leading-5 text-text placeholder-gray-500 focus:outline-none focus:bg-background focus:border-primary focus:ring-1 focus:ring-primary sm:text-sm transition-all"
            />
            {searchQuery && (
              <div className="absolute right-2 top-2">
                <span className="text-xs bg-border px-1.5 py-0.5 rounded text-muted">Enter</span>
              </div>
            )}
          </div>

          <button
            onClick={handleSearch}
            className="px-4 py-2 bg-primary hover:bg-blue-600 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-blue-900/20 flex items-center gap-2"
          >
            Search
          </button>

          <div className="flex bg-surface rounded-lg p-1 border border-border">
            <button
              onClick={() => setSearchMode(SearchMode.EXACT)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                searchMode === SearchMode.EXACT ? 'bg-border text-white shadow-sm' : 'text-muted hover:text-text'
              }`}
            >
              Exact
            </button>
            <button
              onClick={() => isOnline && setSearchMode(SearchMode.HYBRID)}
              disabled={!isOnline}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1 ${
                searchMode === SearchMode.HYBRID
                  ? 'bg-gradient-to-r from-primary to-secondary text-white shadow-sm'
                  : !isOnline
                  ? 'text-muted/50 cursor-not-allowed'
                  : 'text-muted hover:text-text'
              }`}
              title={!isOnline ? 'Unavailable offline' : 'Enable Semantic AI Search'}
            >
              <ZapIcon className="w-3 h-3" /> AI Hybrid
            </button>
          </div>
        </div>

        {/* Search Results */}
        {searchQuery && searchResults.length > 0 && (
          <div className="bg-background/95 backdrop-blur border-b border-border p-4 max-h-64 overflow-y-auto shadow-2xl z-10">
            <div className="text-xs uppercase font-bold text-muted mb-2 tracking-wider">Top Results</div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {searchResults.map(result => (
                <SearchResultItem
                  key={result.note.id}
                  result={result}
                  onClick={() => {
                    setActiveNoteId(result.note.id);
                    setSearchQuery('');
                    setSearchResults([]);
                    setViewMode('editor');
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Workspace */}
        <div className="flex-1 overflow-y-auto relative">{renderContent()}</div>
      </div>

      <ConfirmationModal
        isOpen={!!moveRequest}
        title="Reorganize Knowledge"
        message={
          <div className="space-y-2">
            <p>Are you sure you want to move this note?</p>
            <div className="flex items-center gap-2 p-3 bg-black/20 rounded border border-white/5">
              <span className="text-gray-400">Note:</span>
              <span className="font-bold text-white">{moveRequest?.noteTitle}</span>
            </div>
            <div className="flex justify-center text-gray-500">↓</div>
            <div className="flex items-center gap-2 p-3 bg-blue-500/10 rounded border border-blue-500/20">
              <span className="text-blue-300">New Cluster:</span>
              <span className="font-bold text-blue-100">{moveRequest?.targetClusterName}</span>
            </div>
          </div>
        }
        onConfirm={executeMove}
        onCancel={() => setMoveRequest(null)}
      />
    </div>
  );
};

export default App;
