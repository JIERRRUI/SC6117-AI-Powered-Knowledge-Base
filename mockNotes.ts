import { Note } from "./types";

/**
 * Mock Notes Dataset (100 notes across 8 topics)
 * Used for testing clustering, search, and scalability
 */

export const generateMockNotes = (): Note[] => {
  const notes: Note[] = [];
  let id = 1;

  // Helper to create a note
  const createNote = (
    title: string,
    content: string,
    tags: string[],
    folder: string
  ): Note => ({
    id: `note-${id++}`,
    title,
    content,
    tags,
    createdAt: new Date(
      Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
    ).toISOString(),
    folder,
  });

  // ===== MACHINE LEARNING (18 notes) =====
  notes.push(
    createNote(
      "Introduction to Neural Networks",
      "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using connectionist approaches.",
      ["ml", "neural-networks", "ai"],
      "Machine Learning"
    ),
    createNote(
      "Transformer Architecture Explained",
      "Transformers are deep learning models that use self-attention mechanisms. They revolutionized NLP by processing entire sequences in parallel instead of sequentially.",
      ["transformers", "nlp", "deep-learning"],
      "Machine Learning"
    ),
    createNote(
      "Understanding Convolutional Neural Networks",
      "CNNs are deep learning architectures designed for processing grid-like data, particularly images. They use convolutional layers to extract spatial features.",
      ["cnn", "computer-vision", "deep-learning"],
      "Machine Learning"
    ),
    createNote(
      "BERT Model and Its Applications",
      "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that achieves state-of-the-art results on many NLP tasks.",
      ["bert", "nlp", "language-model"],
      "Machine Learning"
    ),
    createNote(
      "GPT Models: From GPT-1 to GPT-4",
      "GPT models are autoregressive language models trained on large text corpora. They demonstrate remarkable few-shot learning capabilities.",
      ["gpt", "language-model", "generative"],
      "Machine Learning"
    ),
    createNote(
      "Reinforcement Learning Fundamentals",
      "Reinforcement learning teaches agents to make decisions by rewarding desired behaviors. Q-learning and policy gradients are common approaches.",
      ["rl", "reinforcement-learning", "agents"],
      "Machine Learning"
    ),
    createNote(
      "Attention Mechanism in Deep Learning",
      "Attention mechanisms allow models to focus on relevant parts of input. Multi-head attention is used in transformers for parallel processing.",
      ["attention", "transformers", "deep-learning"],
      "Machine Learning"
    ),
    createNote(
      "Transfer Learning and Fine-tuning",
      "Transfer learning reuses pre-trained models for new tasks, significantly reducing training time and data requirements.",
      ["transfer-learning", "fine-tuning", "training"],
      "Machine Learning"
    ),
    createNote(
      "Generative Adversarial Networks (GANs)",
      "GANs consist of two neural networks competing: a generator creates fake data while a discriminator tries to distinguish real from fake.",
      ["gan", "generative", "deep-learning"],
      "Machine Learning"
    ),
    createNote(
      "Recurrent Neural Networks and LSTMs",
      "RNNs process sequential data by maintaining hidden states. LSTMs address the vanishing gradient problem with gate mechanisms.",
      ["rnn", "lstm", "sequence"],
      "Machine Learning"
    ),
    createNote(
      "Embeddings and Word2Vec",
      "Word embeddings represent words as dense vectors in semantic space. Word2Vec uses skip-gram and CBOW architectures.",
      ["embeddings", "nlp", "word2vec"],
      "Machine Learning"
    ),
    createNote(
      "Feature Engineering Best Practices",
      "Feature engineering creates meaningful input features for models. Domain knowledge and experimentation are crucial.",
      ["feature-engineering", "preprocessing", "ml"],
      "Machine Learning"
    ),
    createNote(
      "Cross-validation and Model Evaluation",
      "Cross-validation estimates model performance by splitting data into train/validation sets. Metrics like F1, AUC guide model selection.",
      ["evaluation", "cross-validation", "metrics"],
      "Machine Learning"
    ),
    createNote(
      "Batch Normalization and Dropout",
      "Batch normalization normalizes layer inputs improving training speed. Dropout randomly disables neurons preventing overfitting.",
      ["regularization", "batch-norm", "dropout"],
      "Machine Learning"
    ),
    createNote(
      "Backpropagation Algorithm",
      "Backpropagation efficiently computes gradients through chain rule. It enables training of deep networks.",
      ["backprop", "gradient-descent", "training"],
      "Machine Learning"
    ),
    createNote(
      "Hyperparameter Tuning and Grid Search",
      "Hyperparameters control learning rate, batch size, architecture. Grid search and random search explore parameter spaces.",
      ["hyperparameters", "tuning", "optimization"],
      "Machine Learning"
    ),
    createNote(
      "Clustering Algorithms: K-means and DBSCAN",
      "K-means partitions data into k clusters. DBSCAN finds density-based clusters without specifying cluster count.",
      ["clustering", "unsupervised", "algorithms"],
      "Machine Learning"
    ),
    createNote(
      "Dimensionality Reduction: PCA and t-SNE",
      "PCA reduces dimensions preserving variance. t-SNE creates meaningful 2D/3D visualizations for high-dimensional data.",
      ["dimensionality-reduction", "pca", "tsne"],
      "Machine Learning"
    )
  );

  // ===== WEB DEVELOPMENT (18 notes) =====
  notes.push(
    createNote(
      "HTML5 Semantic Elements",
      "HTML5 provides semantic tags like <article>, <section>, <nav> improving code clarity and SEO.",
      ["html", "web", "semantics"],
      "Web Development"
    ),
    createNote(
      "CSS Flexbox Layout",
      "Flexbox provides flexible one-dimensional layouts for UI components. Properties like flex-direction, justify-content control layout.",
      ["css", "flexbox", "layout"],
      "Web Development"
    ),
    createNote(
      "CSS Grid for Complex Layouts",
      "CSS Grid enables two-dimensional layouts with rows and columns. It's powerful for creating responsive designs.",
      ["css", "grid", "layout"],
      "Web Development"
    ),
    createNote(
      "Responsive Web Design Principles",
      "RWD creates websites that adapt to different screen sizes. Mobile-first design and media queries are key techniques.",
      ["responsive", "mobile", "design"],
      "Web Development"
    ),
    createNote(
      "JavaScript Event Handling",
      "Event handlers respond to user actions like clicks and typing. Event delegation improves performance for dynamic content.",
      ["javascript", "events", "dom"],
      "Web Development"
    ),
    createNote(
      "Async/Await and Promises",
      "Promises represent eventual completion of async operations. Async/await provides cleaner syntax than callback chains.",
      ["javascript", "async", "promises"],
      "Web Development"
    ),
    createNote(
      "REST API Design Best Practices",
      "RESTful APIs use HTTP methods on resources. Proper status codes, versioning, and pagination ensure scalability.",
      ["api", "rest", "backend"],
      "Web Development"
    ),
    createNote(
      "Web Security: CSRF and XSS Prevention",
      "CSRF tokens prevent cross-site requests. Input validation and output encoding prevent XSS attacks.",
      ["security", "web", "vulnerabilities"],
      "Web Development"
    ),
    createNote(
      "Content Security Policy (CSP)",
      "CSP headers restrict resource loading preventing injection attacks. It's a crucial defense-in-depth security measure.",
      ["security", "headers", "csp"],
      "Web Development"
    ),
    createNote(
      "Browser Performance Optimization",
      "Minimize render-blocking resources, optimize images, and implement lazy loading for fast page loads.",
      ["performance", "optimization", "web"],
      "Web Development"
    ),
    createNote(
      "HTTP/2 and HTTP/3 Protocols",
      "HTTP/2 introduces multiplexing and server push. HTTP/3 uses QUIC for improved performance.",
      ["http", "protocols", "networking"],
      "Web Development"
    ),
    createNote(
      "Service Workers and Progressive Web Apps",
      "Service workers enable offline functionality and push notifications. PWAs provide app-like experiences.",
      ["pwa", "offline", "service-workers"],
      "Web Development"
    ),
    createNote(
      "WebSockets for Real-time Communication",
      "WebSockets maintain persistent connections for bidirectional communication, enabling real-time features.",
      ["websockets", "real-time", "communication"],
      "Web Development"
    ),
    createNote(
      "DOM Manipulation and Performance",
      "Batch DOM updates and use DocumentFragment to minimize reflows. Virtual DOM in frameworks provides abstraction.",
      ["dom", "javascript", "performance"],
      "Web Development"
    ),
    createNote(
      "Cookie Management and Same-Site Attribute",
      "Cookies store client-side data. SameSite attribute prevents CSRF attacks by restricting cross-site access.",
      ["cookies", "security", "web"],
      "Web Development"
    ),
    createNote(
      "Web Accessibility (WCAG) Standards",
      "WCAG ensures websites are accessible to people with disabilities. Semantic HTML and ARIA labels are essential.",
      ["accessibility", "wcag", "a11y"],
      "Web Development"
    ),
    createNote(
      "Caching Strategies: Browser and Server",
      "Browser caching stores assets locally. Server-side caching (Redis) accelerates database queries.",
      ["caching", "performance", "optimization"],
      "Web Development"
    ),
    createNote(
      "GraphQL vs REST Architectures",
      "GraphQL allows clients to request specific fields reducing over-fetching. REST is simpler but less flexible.",
      ["graphql", "api", "architecture"],
      "Web Development"
    )
  );

  // ===== PYTHON PROGRAMMING (16 notes) =====
  notes.push(
    createNote(
      "Python List Comprehensions",
      "List comprehensions create lists concisely. They're faster than loops and more Pythonic.",
      ["python", "syntax", "performance"],
      "Python Programming"
    ),
    createNote(
      "Python Decorators and Metaprogramming",
      "Decorators modify function behavior without changing source. They enable elegant cross-cutting concerns.",
      ["python", "decorators", "advanced"],
      "Python Programming"
    ),
    createNote(
      "Context Managers and With Statements",
      "Context managers ensure cleanup with __enter__ and __exit__. The 'with' statement provides automatic resource management.",
      ["python", "context-managers", "file-handling"],
      "Python Programming"
    ),
    createNote(
      "Python Generators and Iterators",
      "Generators produce values on-the-fly using 'yield'. They're memory-efficient for large datasets.",
      ["python", "generators", "iterators"],
      "Python Programming"
    ),
    createNote(
      "Virtual Environments and Dependency Management",
      "venv creates isolated Python environments. pip and poetry manage package dependencies.",
      ["python", "environments", "pip"],
      "Python Programming"
    ),
    createNote(
      "Type Hints and MyPy",
      "Type hints improve code clarity and enable static analysis. MyPy catches type errors before runtime.",
      ["python", "typing", "mypy"],
      "Python Programming"
    ),
    createNote(
      "NumPy Array Operations",
      "NumPy provides efficient numerical computing with n-dimensional arrays. Vectorization avoids slow Python loops.",
      ["python", "numpy", "data-science"],
      "Python Programming"
    ),
    createNote(
      "Pandas DataFrames and Data Manipulation",
      "Pandas DataFrames provide SQL-like operations on tabular data. Groupby and merge enable complex data transformations.",
      ["python", "pandas", "data-science"],
      "Python Programming"
    ),
    createNote(
      "Functional Programming: Map, Filter, Reduce",
      "Functional programming treats computation as function evaluation. These higher-order functions transform sequences.",
      ["python", "functional", "programming"],
      "Python Programming"
    ),
    createNote(
      "Exception Handling Best Practices",
      "Specific exception catching and try-finally blocks ensure robust error handling. Custom exceptions improve code clarity.",
      ["python", "exceptions", "error-handling"],
      "Python Programming"
    ),
    createNote(
      "Python Standard Library Modules",
      "collections, itertools, functools provide powerful abstractions. json, csv, sqlite3 handle common data formats.",
      ["python", "stdlib", "modules"],
      "Python Programming"
    ),
    createNote(
      "Multithreading and Multiprocessing",
      "Threading handles I/O-bound concurrency. Multiprocessing parallelizes CPU-bound tasks bypassing the GIL.",
      ["python", "concurrency", "threading"],
      "Python Programming"
    ),
    createNote(
      "Unit Testing with Pytest",
      "Pytest provides simple assertions and powerful fixtures. Mocking external dependencies ensures isolated tests.",
      ["python", "testing", "pytest"],
      "Python Programming"
    ),
    createNote(
      "Python Package Distribution",
      "setuptools creates installable packages. PyPI hosts packages for pip installation.",
      ["python", "packaging", "distribution"],
      "Python Programming"
    ),
    createNote(
      "Logging in Python Applications",
      "The logging module provides structured logging. Log levels (DEBUG, INFO, ERROR) control verbosity.",
      ["python", "logging", "debugging"],
      "Python Programming"
    ),
    createNote(
      "Regular Expressions in Python",
      "The re module provides pattern matching and text processing. Named groups and lookaheads enable complex matching.",
      ["python", "regex", "text-processing"],
      "Python Programming"
    )
  );

  // ===== DEVOPS & CLOUD (14 notes) =====
  notes.push(
    createNote(
      "Docker Containerization Basics",
      "Docker packages applications and dependencies in containers. Dockerfile defines container images reproducibly.",
      ["docker", "containers", "devops"],
      "DevOps"
    ),
    createNote(
      "Kubernetes Orchestration",
      "Kubernetes manages containerized applications at scale. Pods, services, and deployments organize containers.",
      ["kubernetes", "orchestration", "devops"],
      "DevOps"
    ),
    createNote(
      "CI/CD Pipelines with GitHub Actions",
      "GitHub Actions automate testing and deployment. Workflows define build, test, and release stages.",
      ["ci-cd", "github-actions", "automation"],
      "DevOps"
    ),
    createNote(
      "Infrastructure as Code with Terraform",
      "Terraform provisions cloud infrastructure through code. State files track resource changes.",
      ["terraform", "iac", "cloud"],
      "DevOps"
    ),
    createNote(
      "Git Workflows and Branching Strategies",
      "Git-flow and trunk-based development organize collaboration. Pull requests enable code review.",
      ["git", "version-control", "workflow"],
      "DevOps"
    ),
    createNote(
      "Monitoring and Logging with ELK Stack",
      "Elasticsearch stores logs, Logstash processes them, Kibana visualizes. This enables observability.",
      ["monitoring", "logging", "elk"],
      "DevOps"
    ),
    createNote(
      "Load Balancing and Scaling",
      "Load balancers distribute traffic across servers. Auto-scaling adjusts capacity based on demand.",
      ["load-balancing", "scalability", "devops"],
      "DevOps"
    ),
    createNote(
      "Database Replication and Backup Strategies",
      "Master-slave replication provides redundancy. Regular backups ensure disaster recovery.",
      ["database", "replication", "backup"],
      "DevOps"
    ),
    createNote(
      "Network Security and Firewalls",
      "Firewalls control traffic between networks. Security groups in cloud platforms define access rules.",
      ["security", "networking", "firewall"],
      "DevOps"
    ),
    createNote(
      "Secrets Management and Encryption",
      "HashiCorp Vault manages sensitive data. TLS encryption protects data in transit.",
      ["security", "secrets", "encryption"],
      "DevOps"
    ),
    createNote(
      "Cloud Provider Comparison: AWS vs Azure vs GCP",
      "AWS offers comprehensive services, Azure integrates with Microsoft, GCP excels in analytics.",
      ["cloud", "aws", "azure"],
      "DevOps"
    ),
    createNote(
      "Serverless Computing and Functions",
      "Lambda functions execute code without managing servers. API Gateway triggers functions on HTTP requests.",
      ["serverless", "lambda", "functions"],
      "DevOps"
    ),
    createNote(
      "Message Queues and Event Streaming",
      "RabbitMQ and Kafka decouple service communication. Event streaming enables real-time data processing.",
      ["messaging", "kafka", "queues"],
      "DevOps"
    ),
    createNote(
      "Disaster Recovery and High Availability",
      "RTO (Recovery Time Objective) and RPO (Recovery Point Objective) guide DR strategies. Failover ensures continuity.",
      ["disaster-recovery", "ha", "resilience"],
      "DevOps"
    )
  );

  // ===== DATA SCIENCE (14 notes) =====
  notes.push(
    createNote(
      "Data Preprocessing and Cleaning",
      "Handle missing values, outliers, and duplicates. Feature scaling normalizes numerical features.",
      ["data-science", "preprocessing", "etl"],
      "Data Science"
    ),
    createNote(
      "Exploratory Data Analysis (EDA)",
      "Visualize distributions, correlations, and patterns. Histograms, scatter plots, and heatmaps reveal insights.",
      ["data-science", "eda", "visualization"],
      "Data Science"
    ),
    createNote(
      "Statistical Hypothesis Testing",
      "t-tests, chi-square, ANOVA determine statistical significance. P-values quantify evidence against null hypothesis.",
      ["statistics", "hypothesis-testing", "data-science"],
      "Data Science"
    ),
    createNote(
      "Regression Analysis: Linear and Logistic",
      "Linear regression predicts continuous values. Logistic regression handles binary classification.",
      ["regression", "statistics", "ml"],
      "Data Science"
    ),
    createNote(
      "Decision Trees and Random Forests",
      "Decision trees partition data into pure subsets. Random forests aggregate multiple trees reducing overfitting.",
      ["trees", "ensemble", "ml"],
      "Data Science"
    ),
    createNote(
      "Support Vector Machines (SVM)",
      "SVMs find optimal hyperplanes separating classes. Kernel trick enables non-linear classification.",
      ["svm", "classification", "ml"],
      "Data Science"
    ),
    createNote(
      "Time Series Analysis and Forecasting",
      "ARIMA models capture temporal dependencies. Prophet handles seasonality and holidays.",
      ["time-series", "forecasting", "data-science"],
      "Data Science"
    ),
    createNote(
      "Anomaly Detection Methods",
      "Isolation forests detect outliers. One-class SVM and autoencoders identify abnormal patterns.",
      ["anomaly-detection", "unsupervised", "ml"],
      "Data Science"
    ),
    createNote(
      "Natural Language Processing Basics",
      "Tokenization, stemming, lemmatization preprocess text. TF-IDF and bag-of-words create feature vectors.",
      ["nlp", "text-processing", "data-science"],
      "Data Science"
    ),
    createNote(
      "Sentiment Analysis Techniques",
      "Lexicon-based methods use word dictionaries. Machine learning models predict sentiment from text.",
      ["nlp", "sentiment", "classification"],
      "Data Science"
    ),
    createNote(
      "Recommendation Systems",
      "Collaborative filtering uses user-item interactions. Content-based systems recommend similar items.",
      ["recommendations", "ml", "systems"],
      "Data Science"
    ),
    createNote(
      "A/B Testing and Experimentation",
      "Control groups compare against treatment groups. Statistical power and sample size determine validity.",
      ["ab-testing", "statistics", "experiments"],
      "Data Science"
    ),
    createNote(
      "Model Interpretability and SHAP",
      "SHAP values explain individual predictions. Feature importance visualizes influential features.",
      ["interpretability", "explainability", "ml"],
      "Data Science"
    ),
    createNote(
      "Data Visualization Libraries: Matplotlib, Seaborn, Plotly",
      "Matplotlib provides low-level plotting. Seaborn creates statistical visualizations. Plotly enables interactive charts.",
      ["visualization", "python", "data-science"],
      "Data Science"
    )
  );

  // ===== REACT DEVELOPMENT (14 notes) =====
  notes.push(
    createNote(
      "React Component Lifecycle",
      "Functional components use hooks replacing lifecycle methods. useEffect handles side effects and cleanup.",
      ["react", "hooks", "components"],
      "React Development"
    ),
    createNote(
      "React State Management with useState",
      "useState creates local component state. Updates re-render the component with new state.",
      ["react", "state", "hooks"],
      "React Development"
    ),
    createNote(
      "Context API for State Sharing",
      "Context reduces prop drilling across component trees. useContext hooks access context values.",
      ["react", "context", "state-management"],
      "React Development"
    ),
    createNote(
      "Redux and Redux Middleware",
      "Redux centralizes state in a store. Reducers handle actions updating state predictably.",
      ["redux", "state-management", "react"],
      "React Development"
    ),
    createNote(
      "Controlled vs Uncontrolled Components",
      "Controlled components derive state from props. Uncontrolled components use refs for direct DOM access.",
      ["react", "forms", "components"],
      "React Development"
    ),
    createNote(
      "React Performance Optimization",
      "useMemo memoizes expensive computations. useCallback prevents unnecessary re-renders of child components.",
      ["react", "performance", "optimization"],
      "React Development"
    ),
    createNote(
      "Code Splitting and Lazy Loading",
      "React.lazy enables code splitting at route boundaries. Dynamic imports reduce initial bundle size.",
      ["react", "performance", "bundling"],
      "React Development"
    ),
    createNote(
      "Error Boundaries in React",
      "Error boundaries catch rendering errors. They prevent entire app crashes and enable graceful degradation.",
      ["react", "error-handling", "components"],
      "React Development"
    ),
    createNote(
      "Form Handling with React Hook Form",
      "React Hook Form minimizes re-renders for form changes. Integration with validation libraries like Zod.",
      ["react", "forms", "validation"],
      "React Development"
    ),
    createNote(
      "Routing with React Router",
      "BrowserRouter enables client-side navigation. Route components render based on URL paths.",
      ["react", "routing", "navigation"],
      "React Development"
    ),
    createNote(
      "Server-Side Rendering with Next.js",
      "Next.js renders React on servers. Static generation and incremental static regeneration improve performance.",
      ["nextjs", "ssr", "react"],
      "React Development"
    ),
    createNote(
      "Testing React Components with React Testing Library",
      "Testing Library encourages testing user behavior. Queries like getByRole simulate real user interactions.",
      ["react", "testing", "jest"],
      "React Development"
    ),
    createNote(
      "Styling Approaches: CSS Modules and Styled Components",
      "CSS Modules scope styles locally. Styled Components write CSS in JavaScript.",
      ["react", "styling", "css"],
      "React Development"
    ),
    createNote(
      "Accessibility in React Applications",
      "Semantic HTML, ARIA labels, and keyboard navigation ensure accessibility. Testing tools like axe-core verify compliance.",
      ["react", "accessibility", "a11y"],
      "React Development"
    )
  );

  // ===== DATABASE DESIGN (10 notes) =====
  notes.push(
    createNote(
      "Relational Database Design and Normalization",
      "Normalization reduces redundancy through 1NF, 2NF, 3NF. Primary and foreign keys maintain referential integrity.",
      ["database", "sql", "design"],
      "Database"
    ),
    createNote(
      "Database Indexing Strategies",
      "Indexes accelerate queries by creating sorted data structures. B-tree and hash indexes serve different purposes.",
      ["database", "indexing", "performance"],
      "Database"
    ),
    createNote(
      "Query Optimization and EXPLAIN Plans",
      "EXPLAIN shows query execution plans. Analysis reveals missing indexes and inefficient joins.",
      ["database", "optimization", "sql"],
      "Database"
    ),
    createNote(
      "Transactions and ACID Properties",
      "ACID ensures data consistency and reliability. Isolation levels control concurrent transaction visibility.",
      ["database", "transactions", "acid"],
      "Database"
    ),
    createNote(
      "NoSQL Databases: MongoDB, DynamoDB, Cassandra",
      "NoSQL provides flexible schemas for unstructured data. Eventually consistent distributed systems scale horizontally.",
      ["nosql", "mongodb", "database"],
      "Database"
    ),
    createNote(
      "Document Databases vs Key-Value Stores",
      "Document databases (MongoDB) store nested structures. Key-value stores (Redis) provide fast lookups.",
      ["nosql", "database", "design"],
      "Database"
    ),
    createNote(
      "Sharding and Horizontal Scaling",
      "Sharding distributes data across multiple servers. Range and hash sharding determine data distribution.",
      ["database", "scaling", "sharding"],
      "Database"
    ),
    createNote(
      "Database Partitioning Strategies",
      "Range partitioning, list partitioning, hash partitioning organize data. Improves query performance and maintenance.",
      ["database", "partitioning", "design"],
      "Database"
    ),
    createNote(
      "Full-Text Search and Elasticsearch",
      "Full-text search indexes documents for keyword searches. Elasticsearch provides distributed search at scale.",
      ["database", "search", "elasticsearch"],
      "Database"
    ),
    createNote(
      "Data Warehousing and OLAP",
      "Data warehouses integrate data from multiple sources. OLAP cubes enable multidimensional analysis.",
      ["database", "analytics", "warehouse"],
      "Database"
    )
  );

  // ===== UI/UX DESIGN (8 notes) =====
  notes.push(
    createNote(
      "Design Systems and Component Libraries",
      "Design systems ensure consistency across products. Storybook documents component variations.",
      ["design", "ux", "components"],
      "Design"
    ),
    createNote(
      "Color Theory and Typography",
      "Color psychology influences user perception. Typography hierarchy guides attention.",
      ["design", "visual", "principles"],
      "Design"
    ),
    createNote(
      "Information Architecture",
      "Sitemaps and wireframes plan navigation structures. Card sorting validates information organization.",
      ["ux", "ia", "design"],
      "Design"
    ),
    createNote(
      "User Research and Personas",
      "User interviews and surveys inform personas. Personas guide design decisions.",
      ["ux", "research", "personas"],
      "Design"
    ),
    createNote(
      "Prototyping Tools and Wireframing",
      "Figma, Sketch, Adobe XD enable collaborative design. Prototypes test interactions before development.",
      ["design", "prototyping", "tools"],
      "Design"
    ),
    createNote(
      "Interaction Design and Microinteractions",
      "Microinteractions provide feedback for actions. Animations guide user attention.",
      ["interaction-design", "ux", "animations"],
      "Design"
    ),
    createNote(
      "Usability Testing Methods",
      "Moderated tests observe users completing tasks. Eye-tracking reveals attention patterns.",
      ["ux", "testing", "research"],
      "Design"
    ),
    createNote(
      "Dark Mode Design Considerations",
      "Dark mode reduces eye strain and battery usage. Contrast ratios must meet accessibility standards.",
      ["design", "accessibility", "ux"],
      "Design"
    )
  );

  // ===== MISCELLANEOUS (2 notes) to reach 100
  notes.push(
    createNote(
      "Technical Writing for Documentation",
      "Clear technical writing explains complex concepts. Active voice and simple language improve readability.",
      ["documentation", "writing", "technical"],
      "Writing"
    ),
    createNote(
      "Software Architecture Patterns: MVC, MVVM, Clean Architecture",
      "MVC separates concerns into Model, View, Controller. Clean Architecture emphasizes testability and maintainability.",
      ["architecture", "design-patterns", "software"],
      "Architecture"
    )
  );

  return notes;
};

/**
 * Generate clustered notes for testing clustering algorithms
 * Returns notes pre-organized into visible clusters for validation
 */
export const generateClusteredMockNotes = () => {
  const allNotes = generateMockNotes();

  return {
    totalNotes: allNotes.length,
    notes: allNotes,
    expectedClusters: [
      {
        name: "Machine Learning",
        noteIds: allNotes
          .filter((n) => n.folder === "Machine Learning")
          .map((n) => n.id),
      },
      {
        name: "Web Development",
        noteIds: allNotes
          .filter((n) => n.folder === "Web Development")
          .map((n) => n.id),
      },
      {
        name: "Python Programming",
        noteIds: allNotes
          .filter((n) => n.folder === "Python Programming")
          .map((n) => n.id),
      },
      {
        name: "DevOps",
        noteIds: allNotes.filter((n) => n.folder === "DevOps").map((n) => n.id),
      },
      {
        name: "Data Science",
        noteIds: allNotes
          .filter((n) => n.folder === "Data Science")
          .map((n) => n.id),
      },
      {
        name: "React Development",
        noteIds: allNotes
          .filter((n) => n.folder === "React Development")
          .map((n) => n.id),
      },
      {
        name: "Database",
        noteIds: allNotes
          .filter((n) => n.folder === "Database")
          .map((n) => n.id),
      },
      {
        name: "Design",
        noteIds: allNotes.filter((n) => n.folder === "Design").map((n) => n.id),
      },
    ],
  };
};

/**
 * Export as loadable test data
 */
export const mockNotesDatabase = {
  allNotes: generateMockNotes(),
  clustered: generateClusteredMockNotes(),
  topicCount: 8,
  totalNotes: 100,
};
