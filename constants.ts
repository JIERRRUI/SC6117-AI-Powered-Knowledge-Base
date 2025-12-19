import { Note } from "./types";

// Constants
export const APP_NAME = "Synapse";

// Sample notes for demo/testing - these can be loaded into the database
export const SAMPLE_NOTES: Note[] = [
  {
    id: "sample-1",
    title: "Machine Learning Basics",
    content:
      "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Key concepts include supervised learning, unsupervised learning, and reinforcement learning. Popular libraries include scikit-learn, TensorFlow, and PyTorch.",
    tags: ["ml", "ai"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/ai",
  },
  {
    id: "sample-2",
    title: "Deep Learning Guide",
    content:
      "Deep learning uses neural networks with multiple layers to model complex patterns. Common architectures include CNNs for images, RNNs for sequences, and Transformers for NLP tasks.",
    tags: ["dl", "ai"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/ai",
  },
  {
    id: "sample-3",
    title: "Chocolate Chip Cookies",
    content:
      "Classic chocolate chip cookie recipe: Mix butter, sugar, eggs, and vanilla. Combine flour, baking soda, and salt. Fold in chocolate chips. Bake at 375°F for 9-11 minutes.",
    tags: ["cooking", "dessert"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/cooking",
  },
  {
    id: "sample-4",
    title: "React Hooks Tutorial",
    content:
      "React Hooks allow function components to use state and lifecycle features. useState manages local state, useEffect handles side effects, useContext accesses context, and useMemo optimizes performance.",
    tags: ["react", "web"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/web",
  },
  {
    id: "sample-5",
    title: "Computer Vision with CNNs",
    content:
      "Convolutional Neural Networks excel at image tasks. They use convolution layers to detect features, pooling to reduce dimensions, and fully connected layers for classification.",
    tags: ["cv", "ai"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/ai",
  },
  {
    id: "sample-6",
    title: "Italian Pasta Carbonara",
    content:
      "Authentic carbonara recipe: Cook guanciale until crispy. Whisk eggs with pecorino romano. Toss hot pasta with guanciale, then quickly mix in egg mixture off heat. Season with black pepper.",
    tags: ["cooking", "italian"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/cooking",
  },
  {
    id: "sample-7",
    title: "Vue.js Component Patterns",
    content:
      "Vue 3 Composition API provides better code organization. Use ref and reactive for state, computed for derived values, and watch for side effects. Props flow down, events flow up.",
    tags: ["vue", "web"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/web",
  },
  {
    id: "sample-8",
    title: "NLP with Transformers",
    content:
      "Transformer architecture revolutionized NLP with self-attention mechanisms. Models like BERT, GPT, and T5 achieve state-of-the-art results on language understanding and generation tasks.",
    tags: ["nlp", "ai"],
    createdAt: new Date().toISOString().split("T")[0],
    folder: "/samples/ai",
  },
];
