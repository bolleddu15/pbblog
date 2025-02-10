document.addEventListener("DOMContentLoaded", () => {
  const tabContents = document.querySelectorAll(".tabs__tab");
  const navLinks = document.querySelectorAll("nav ul li a");
  const projectContent = document.getElementById("project-content");
  const publicationContent = document.getElementById("publication-content");
  const storyContent = document.getElementById("story-content");

  // Function to remove active class and set the active tab
  function setActiveTab(tabId) {
    tabContents.forEach(tab => tab.classList.remove("active"));
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
      targetTab.classList.add("active");
    }
    // Update active class on navigation links (only for main tabs)
    navLinks.forEach(link => {
      const dataTab = link.getAttribute("data-tab");
      if (dataTab === tabId) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });
  }

  // Function to load project details
  function loadProject(projectId) {
    const projectData = {
      "project1": `
            <h2><b>Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls</b></h2>
            <p>Our investigation focuses on identifying and addressing inherent biases in these multimodal analyses. Bias in this context can distort market perception and decision-making, ultimately affecting investor trust and financial stability.</p>
            <h3><b>Key Definitions</b></h3>
            <p><strong>Multimodal Analysis:</strong> The integration of multiple data sources—such as audio, visual, and textual inputs—to create a comprehensive understanding of complex communications.</p>
            <p><strong>Financial Earnings Calls:</strong> Periodic teleconferences where publicly-traded companies discuss their financial performance, answer analyst queries, and provide strategic guidance.</p>
            <p><strong>Bias (in AI and Data Analysis):</strong> Systematic errors or prejudices in data processing and algorithmic interpretation that can lead to skewed results.</p>
            <h3><b>Algorithm</b></h3>
            <p>A set of rules or instructions used to process data and make predictions or classifications.</p>
            <h3><b>Problem Statement</b></h3>
            <p>Despite the advancements in multimodal analysis, bias remains a persistent challenge:</p>
            <ul>
              <li><strong>Data Imbalance:</strong> Earnings calls often reflect a diversity of communication styles, languages, and accents.</li>
              <li><strong>Interpretation Ambiguity:</strong> Visual cues and tone of voice can be misinterpreted, leading to oversimplified sentiment analysis.</li>
              <li><strong>Algorithmic Fairness:</strong> Without rigorous bias detection, these algorithms may propagate or amplify existing prejudices.</li>
            </ul>
            <p><strong>Our mission:</strong> Develop a robust, fair, and transparent multimodal analysis framework that detects, quantifies, and mitigates biases in financial earnings call evaluations.</p>
            <h3><b>Methodology and Unique Approach</b></h3>
            <h4><b>Data Aggregation & Preprocessing</b></h4>
            <ul>
              <li><strong>Multimodal Data Collection:</strong> Gather audio, video, and transcript data from diverse earnings calls.</li>
              <li><strong>Normalization:</strong> Standardize and clean data to mitigate input-level biases.</li>
            </ul>
            <h4><b>Feature Extraction</b></h4>
            <ul>
              <li><strong>Audio:</strong> Analyze tone, pitch, and inflection.</li>
              <li><strong>Visual:</strong> Leverage computer vision techniques to decode facial expressions.</li>
              <li><strong>Text:</strong> Use NLP to capture sentiment and contextual nuances.</li>
            </ul>
            <h4><b>Bias Detection Module</b></h4>
            <ul>
              <li>Implement fairness-aware machine learning algorithms to detect discrepancies.</li>
              <li>Incorporate real-time bias metrics for transparency.</li>
            </ul>
            <h4><b>Algorithmic Fusion</b></h4>
            <p>Integrate insights from each modality through ensemble methods to generate a holistic view. Utilize explainable AI (XAI) techniques for interpretability.</p>
            <h4><b>Validation and Iteration</b></h4>
            <ul>
              <li>Continuously validate against benchmark datasets.</li>
              <li>Engage with domain experts for iterative feedback.</li>
            </ul>
          `,
      "project2": `
            <p>Customer segmentation enables data-driven marketing, leading to better engagement and profitability.</p>
            <h2><b>Enhanced Customer Segmentation with RFM and Behavioral Analysis</b></h2>
            <p>By leveraging RFM (Recency, Frequency, Monetary) analysis and behavioral data, organizations can achieve targeted marketing that drives higher engagement and conversion rates.</p>
            <h3><b>Foundations of Segmentation</b></h3>
            <h4><b>Customer Segmentation</b></h4>
            <p><strong>Key Objectives:</strong></p>
            <ul>
              <li>Personalize marketing efforts.</li>
              <li>Optimize resource allocation.</li>
              <li>Enhance customer lifetime value.</li>
            </ul>
            <h4><b>Technographic Segmentation</b></h4>
            <p><strong>Key Objectives:</strong></p>
            <ul>
              <li>Identify technology adoption trends.</li>
              <li>Tailor solutions to fit the technological maturity of customers.</li>
              <li>Optimize digital marketing channels.</li>
            </ul>
            <h3><b>RFM Analysis</b></h3>
            <p>RFM analysis is a data-driven method used to evaluate customers based on three key metrics:</p>
            <ul>
              <li><strong>Recency (R):</strong> How recently a customer made a purchase.</li>
              <li><strong>Frequency (F):</strong> How often the customer makes a purchase.</li>
              <li><strong>Monetary (M):</strong> How much the customer spends.</li>
            </ul>
            <h3><b>Algorithmic Approach</b></h3>
            <ul>
              <li><strong>Data Collection:</strong> Gather historical transaction data.</li>
              <li><strong>Scoring:</strong> Assign scores for each R, F, and M metric.</li>
              <li><strong>Composite Score:</strong> Combine the scores to create an overall RFM score.</li>
              <li><strong>Segmentation:</strong> Classify customers into segments such as “Champions,” “Loyal Customers,” “At Risk,” etc.</li>
            </ul>
            <h3><b>Behavioral Analysis in Marketing</b></h3>
            <p><strong>Definition:</strong> Examines actions and patterns of customers, including browsing habits and content consumption.</p>
            <p><strong>Key Metrics:</strong></p>
            <ul>
              <li>Engagement Rate.</li>
              <li>Click-Through Rate (CTR).</li>
              <li>Time Spent on product or content.</li>
            </ul>
            <h3><b>Algorithms & Techniques in Segmentation</b></h3>
            <h4><b>Clustering Algorithms</b></h4>
            <p><strong>Definition:</strong> Grouping data points based on similarity.</p>
            <p><strong>Common Algorithms:</strong></p>
            <ul>
              <li>
                <strong>K-Means Clustering:</strong>
                <ul>
                  <li><strong>Process:</strong> Iteratively assigns points to clusters.</li>
                  <li><strong>Application:</strong> Segmenting customers based on RFM scores.</li>
                  <li><strong>Consideration:</strong> Requires a predefined number of clusters.</li>
                </ul>
              </li>
              <li>
                <strong>Hierarchical Clustering:</strong>
                <ul>
                  <li><strong>Process:</strong> Builds a hierarchy of clusters using agglomerative or divisive approaches.</li>
                  <li><strong>Application:</strong> Useful when the number of clusters is not known.</li>
                  <li><strong>Consideration:</strong> Can be computationally expensive.</li>
                </ul>
              </li>
              <li>
                <strong>DBSCAN:</strong>
                <ul>
                  <li><strong>Process:</strong> Identifies clusters based on density.</li>
                  <li><strong>Application:</strong> Effective in detecting outliers in customer behavior data.</li>
                  <li><strong>Consideration:</strong> Sensitive to parameters like epsilon and minimum points.</li>
                </ul>
              </li>
            </ul>
            <h4><b>Association Rule Mining</b></h4>
            <p><strong>Definition:</strong> Discovers interesting associations between variables in large datasets.</p>
            <p><strong>Example:</strong> Apriori Algorithm for cross-selling and upselling.</p>
            <h4><b>Predictive Analytics & Machine Learning</b></h4>
            <p><strong>Definition:</strong> Leveraging historical data to predict future customer behaviors.</p>
            <p><strong>Techniques:</strong></p>
            <ul>
              <li><strong>Regression Analysis:</strong> Predicts continuous outcomes.</li>
              <li><strong>Classification Algorithms:</strong> Predicts categorical outcomes.</li>
            </ul>
          `,
      "project3": `
            <h2><b>Context-Aware Healthcare Chatbot using RAG & Knowledge Graphs</b></h2>
            <p><strong>Problem Statement:</strong> Traditional healthcare chatbots often fail to provide accurate, context-aware responses due to reliance on predefined intents. This project enhances chatbot intelligence using Retrieval-Augmented Generation (RAG) and knowledge graphs to provide precise, real-time medical responses.</p>
            <h3><b>Algorithms & Architectures Used:</b></h3>
            <ul>
              <li><b>LangChain for Context-Aware Conversations:</b> Utilizes large language models for dynamic query understanding.</li>
              <li><b>Neo4j Graph Database:</b> Stores relationships between medical entities for semantic search.</li>
              <li><b>RAG (Retrieval-Augmented Generation):</b> Combines dense retrieval with generative models for accurate responses.</li>
              <li><b>Named Entity Recognition (NER) & Entity Linking:</b> Identifies and links medical terms.</li>
              <li><b>Prompt Engineering & Vector Indexing:</b> Optimizes query responses by retrieving context from Neo4j.</li>
            </ul>
            <h3><b>Reasons for Choosing These Approaches:</b></h3>
            <ul>
              <li>Knowledge graphs ensure structured medical reasoning.</li>
              <li>RAG models mitigate LLM hallucinations by grounding responses in real-time data.</li>
              <li>Vector search techniques enhance medical information retrieval.</li>
              <li>NLP-powered entity recognition enables context-driven insights.</li>
            </ul>
            <h3><b>Challenges Faced & Solutions:</b></h3>
            <p>Addressed model biases, optimized query efficiency, and integrated privacy-preserving mechanisms for sensitive healthcare data.</p>
          `,
      "project4": `
            <h2><b>Federated Learning for Multi-Class Radiology Image Analysis</b></h2>
            <p><strong>Problem Statement:</strong> Automating multi-class radiology image analysis across hospitals faces issues like inconsistent labeling, data silos, and privacy concerns. Traditional CNN-based models require centralized data, posing privacy risks and model bias. This project leverages Federated Learning (FL) to train CNN models without sharing patient data across hospitals.</p>
            <h3><b>Algorithms & Architectures Used:</b></h3>
            <ul>
              <li><b>Convolutional Neural Networks (CNNs – ResNet, EfficientNet):</b> Extracts features from multi-panel radiology scans (e.g., MRI, CT, X-ray).</li>
              <li><b>Federated Learning with Differential Privacy:</b> Secure, decentralized training without exposing raw data.</li>
              <li><b>Self-Supervised Learning (SimCLR, BYOL):</b> Improves label efficiency by pretraining on unlabeled medical images.</li>
              <li><b>Attention-Based Transformers (Swin Transformer, ViTs):</b> Enhances CNN performance by capturing long-range dependencies.</li>
              <li><b>Multi-View Image Fusion:</b> Integrates multiple scan views to enhance classification accuracy.</li>
            </ul>
            <h3><b>Reasons for Choosing These Approaches:</b></h3>
            <ul>
              <li>Federated Learning preserves patient privacy while enabling cross-hospital collaboration.</li>
              <li>Multi-Panel Classification ensures comprehensive image analysis.</li>
              <li>Self-Supervised Learning minimizes annotation costs.</li>
              <li>Transformers improve feature extraction in high-resolution scans.</li>
            </ul>
          `
    };

    projectContent.innerHTML = projectData[projectId] || "<p>No project found.</p>";
    setActiveTab("tab_project");
  }

  // Function to load publication details
  function loadPublication(publicationId) {
    const publicationData = {
      "pub1": `<div style="text-align: center;">
  <h2>An Analysis of Clustering Techniques.</h2>
  <p class="project-description">
    The analysis and comparative study of these clustering techniques contribute to the development of more robust, scalable, and interpretable clustering algorithms. By addressing common challenges such as high dimensionality and noise, the work paves the way for improved performance in numerous real-world applications, thereby influencing subsequent research and practical implementations in data mining and pattern recognition.
  </p>
  <h3>Algorithms and Methodologies</h3>
  <h4>2.1 K-Means Clustering</h4>
  <p><strong>Need:</strong><br>
    K-Means offers a simple yet effective partitioning strategy, especially suitable for large datasets where computational efficiency is paramount.
  </p>
  <p><strong>Algorithm:</strong><br>
    K-Means partitions data into <em>K</em> clusters by minimizing the within-cluster sum of squares (WCSS).<br>
    <strong>Initialization:</strong> Choose <em>K</em> centroids (often using k-means++ for improved convergence).<br>
    <strong>Assignment:</strong> Assign each data point <em>x</em> to the nearest centroid <em>μ<sub>i</sub></em>.<br>
    <strong>Update:</strong> Recalculate centroids as the mean of the assigned points.<br>
    <strong>Iteration:</strong> Repeat until convergence.
  </p>
  <p><strong>Mathematical Equation:</strong><br>
    The objective function to minimize is:
  </p>
  <p style="text-align: center;">J = &Sigma;<sub>i=1</sub><sup>K</sup> &Sigma;<sub>x ∈ C<sub>i</sub></sub> ||x − μ<sub>i</sub>||<sup>2</sup></p>
  <p>
    where <em>C<sub>i</sub></em> is the set of points assigned to cluster <em>i</em> and <em>μ<sub>i</sub></em> is the centroid of cluster <em>i</em>.
  </p>
  <h4>2.2 Hierarchical Clustering</h4>
  <p><strong>Need:</strong><br>
    Hierarchical methods provide a multilevel clustering framework, ideal for discovering nested clusters without a predetermined number of clusters.
  </p>
  <p><strong>Algorithm:</strong><br>
    Two primary approaches exist: agglomerative (bottom-up) and divisive (top-down).
  </p>
  <p><strong>Agglomerative Clustering:</strong><br>
    Start with each data point as its own cluster.<br>
    Iteratively merge the closest clusters using linkage criteria (e.g., single, complete, or average linkage).
  </p>
  <p><strong>Mathematical Formulation (Single Linkage):</strong><br>
    For clusters <em>C<sub>i</sub></em> and <em>C<sub>j</sub></em>, the distance is:
  </p>
  <p style="text-align: center;">d(C<sub>i</sub>, C<sub>j</sub>) = min<sub>x ∈ C<sub>i</sub>, y ∈ C<sub>j</sub></sub> ||x − y||</p>
  <h4>2.3 DBSCAN</h4>
  <p><strong>Need:</strong><br>
    DBSCAN is designed to identify clusters of arbitrary shape and is robust to noise, making it highly effective in real-world applications with irregular cluster structures.
  </p>
  <p><strong>Algorithm:</strong><br>
    <strong>Core Points:</strong> A point <em>x</em> is a core point if at least <em>minPts</em> points are within its <em>ε</em>-neighborhood:<br>
    N<sub>ε</sub>(x) = { y ∈ D : ||x − y|| ≤ ε } with |N<sub>ε</sub>(x)| ≥ minPts.
  </p>
  <p><strong>Expansion:</strong> Clusters are formed by connecting core points and their reachable neighbors.</p>
  <h4>2.4 Expectation-Maximization (EM) Clustering for Gaussian Mixture Models</h4>
  <p><strong>Need:</strong><br>
    EM clustering allows for soft assignments of points to clusters and is well-suited for data that can be modeled as a mixture of distributions.
  </p>
  <p><strong>Algorithm:</strong><br>
    <strong>E-Step:</strong> Compute the probability that each point belongs to each Gaussian component.<br>
    <strong>M-Step:</strong> Update the parameters of each Gaussian component (mean, covariance, and mixing coefficients) to maximize the likelihood of the observed data.
  </p>
  <p><strong>Mathematical Formulation:</strong><br>
    For a data point <em>x</em> and a mixture of <em>K</em> Gaussians, the likelihood is:
  </p>
  <p style="text-align: center;">p(x) = &Sigma;<sub>i=1</sub><sup>K</sup> π<sub>i</sub> N(x | μ<sub>i</sub>, Σ<sub>i</sub>)</p>
  <p>
    where <em>π<sub>i</sub></em> are the mixing coefficients, and N(x | μ<sub>i</sub>, Σ<sub>i</sub>) is the Gaussian density function.
  </p>
</div>`,
      "pub2": `<div style="text-align: center;">
  <h2>Hadoop: An Overview of Data Security</h2>
  <p class="project-description">
    In the era of big data, the Hadoop ecosystem has emerged as a cornerstone for processing and storing vast amounts of information. However, the distributed nature of Hadoop also introduces significant security challenges. This paper provides an in-depth overview of data security mechanisms in Hadoop, focusing on authentication, encryption, and access control. Key security algorithms such as Kerberos for authentication and encryption algorithms like RSA and AES for data protection are discussed alongside their mathematical formulations. The impact of these security measures on ensuring data integrity and privacy in distributed systems is also examined.
  </p>
  <h3>1. Introduction</h3>
  <p>
    Hadoop is a widely adopted framework for distributed storage and processing of large datasets. Given the sensitivity of data managed in Hadoop Distributed File System (HDFS) and MapReduce computations, robust security measures are essential. The need for stringent security protocols is driven by the increasing number of cyber threats targeting big data infrastructures. This work outlines the key components of Hadoop security and provides a detailed analysis of the algorithms and protocols used to secure data within the ecosystem.
  </p>
  <h3>2. Security Framework and Methodologies</h3>
  <h4>2.1 Authentication via Kerberos</h4>
  <p><strong>Need:</strong><br>
    Kerberos provides a secure authentication mechanism based on ticket-granting systems, ensuring that only verified users and services gain access to the Hadoop cluster.
  </p>
  <p><strong>Algorithm:</strong><br>
    <strong>Ticket Granting:</strong> A client obtains a ticket from the Authentication Server (AS) and then a service ticket from the Ticket Granting Server (TGS) to access a specific service.<br>
    <strong>Mutual Authentication:</strong> Both the client and server verify each other’s identities using these time-stamped tickets.
  </p>
  <p><strong>Technical Keywords:</strong><br>
    Kerberos, Ticket Granting Ticket (TGT), Mutual Authentication, Key Distribution Center (KDC).
  </p>
  <h4>2.2 Encryption Techniques</h4>
  <p><strong>Need:</strong><br>
    Encryption ensures that data remains confidential and secure both at rest and in transit, protecting sensitive information from unauthorized access.
  </p>
  <p><strong>Algorithms Implemented:</strong><br>
    RSA Encryption: An asymmetric encryption algorithm used primarily for secure key exchange.<br>
    AES (Advanced Encryption Standard): A symmetric encryption algorithm widely used for encrypting large volumes of data.
  </p>
  <p><strong>Mathematical Formulations:</strong></p>
  <p><strong>RSA Encryption:</strong></p>
  <p><strong>Key Generation:</strong><br>
    Choose two large prime numbers <em>p</em> and <em>q</em>; compute <em>n = p × q</em>.<br>
    Compute Euler’s totient: <em>φ(n) = (p − 1)(q − 1)</em>.<br>
    Choose a public exponent <em>e</em> such that gcd(e, φ(n)) = 1.<br>
    Compute the private exponent <em>d</em> as the modular inverse of <em>e</em> modulo <em>φ(n)</em>:<br>
    <em>d ≡ e<sup>−1</sup> mod φ(n)</em>.
  </p>
  <p><strong>Encryption:</strong><br>
    For a message <em>m</em>, the ciphertext <em>c</em> is computed as: <em>c = m<sup>e</sup> mod n</em>.
  </p>
  <p><strong>Decryption:</strong><br>
    Recover the original message: <em>m = c<sup>d</sup> mod n</em>.
  </p>
  <p><strong>AES Encryption:</strong><br>
    While AES involves complex key schedules and substitution–permutation network transformations, its primary operation is based on multiple rounds of byte substitution, row and column transformations, and mixing operations. Its security lies in the diffusion and confusion properties ensured by these rounds.
  </p>
  <h4>2.3 Access Control Lists (ACLs)</h4>
  <p><strong>Need:</strong><br>
    ACLs provide fine-grained control over who can access various data resources within the Hadoop ecosystem, ensuring that only authorized users can read, write, or modify sensitive data.
  </p>
</div>`,
      "pub3": `<div style="text-align: center;">
  <h2>Neuro-Symbolic Large Action Models: A Unified Theoretical Framework for Generalizable AI Across Vision, Language, and Robotics</h2>
  <p class="project-description">
    The integration of neural networks with symbolic reasoning has the potential to create AI systems that are both powerful and interpretable. This paper presents a unified theoretical framework for neuro-symbolic large action models that generalize across multiple modalities, including vision, language, and robotics. The framework synergizes deep learning for feature extraction with symbolic reasoning for high-level decision making. Detailed mathematical formulations—such as a unified loss function and attention mechanisms—are provided to support the framework’s design. This approach addresses the limitations of purely statistical models, thereby enhancing generalizability and interpretability in complex, multi-modal environments.
  </p>
  <h3>1. Introduction</h3>
  <p>
    Contemporary AI systems often struggle with the challenge of generalizing across disparate domains. Purely neural models excel at pattern recognition but lack explicit reasoning, while symbolic models provide logical inference but struggle with raw sensory input. This publication introduces a neuro-symbolic framework designed to bridge this gap, offering a scalable and interpretable model that combines the strengths of both paradigms.
  </p>
  <h3>2. Framework and Methodologies</h3>
  <h4>2.1 Neural Components</h4>
  <p><strong>Need:</strong><br>
    Deep neural networks (DNNs) are crucial for extracting rich features from high-dimensional data, such as images and text.
  </p>
  <p><strong>Algorithms:</strong><br>
    Convolutional Neural Networks (CNNs) for vision.<br>
    Recurrent Neural Networks (RNNs) or Transformers for language.<br>
    Graph Neural Networks (GNNs) for capturing relationships in robotic action planning.
  </p>
  <h4>2.2 Symbolic Reasoning Modules</h4>
  <p><strong>Need:</strong><br>
    Symbolic reasoning provides explicit, interpretable logic for decision-making and planning.
  </p>
  <p><strong>Approach:</strong><br>
    Rule-based systems and logical inference engines that interact with the neural components to provide high-level action strategies.
  </p>
  <h4>2.3 Hybrid Integration via Attention Mechanisms</h4>
  <p><strong>Algorithm:</strong><br>
    Attention mechanisms are employed to dynamically weigh the contributions of different modalities during decision making.
  </p>
  <p><strong>Mathematical Formulation:</strong><br>
    A unified objective function is formulated as:
  </p>
  <p style="text-align: center;">min<sub>θ,ϕ</sub> L(θ,ϕ) = L<sub>vision</sub>(θ) + L<sub>language</sub>(ϕ) + λ L<sub>symbolic</sub>(θ,ϕ)</p>
  <p>
    where <em>θ</em> and <em>ϕ</em> denote the parameters of the neural and symbolic components, respectively, and <em>λ</em> balances their contributions.
  </p>
  <p><strong>Attention Mechanism Equation:</strong><br>
    The scaled dot-product attention is defined as:
  </p>
  <p style="text-align: center;">Attention(Q, K, V) = softmax( QK<sup>T</sup> / √d<sub>k</sub> ) V</p>
  <p>
    where <em>Q</em>, <em>K</em>, and <em>V</em> represent the query, key, and value matrices, respectively, and <em>d<sub>k</sub></em> is the dimensionality of the key vectors.
  </p>
</div>`,
      "pub4": `<div style="text-align: center;">
  <h2>Self-Sustaining Artificial Generative Systems: Exploring Autonomous Meta-Learning and Cross-Domain Knowledge Fusion</h2>
  <p class="project-description">
    Autonomous generative systems that can learn and adapt without constant human intervention are a key frontier in artificial intelligence. This paper explores the design of self-sustaining artificial generative systems through the integration of autonomous meta-learning and cross-domain knowledge fusion. By combining state-of-the-art algorithms such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Model-Agnostic Meta-Learning (MAML), the proposed system continuously refines its performance across multiple domains. Detailed mathematical formulations of the GAN adversarial loss, the VAE evidence lower bound (ELBO), and MAML’s gradient-based update rules are presented, showcasing the technical rigor behind the approach.
  </p>
  <h3>1. Introduction</h3>
  <p>
    Generative models have traditionally been static, requiring retraining when applied to new tasks. The need for self-sustaining systems—capable of autonomous adaptation and cross-domain learning—is increasingly recognized as essential for the next generation of AI. This publication addresses this gap by proposing a framework that leverages meta-learning to facilitate rapid adaptation and knowledge fusion across diverse domains.
  </p>
  <h3>2. Methodologies and Algorithms</h3>
  <h4>2.1 Generative Adversarial Networks (GANs)</h4>
  <p><strong>Need:</strong><br>
    GANs generate high-quality synthetic data by pitting a generator against a discriminator in a minimax game.
  </p>
  <p><strong>Mathematical Formulation:</strong><br>
    The GAN objective is defined as:
  </p>
  <p style="text-align: center;">min<sub>G</sub> max<sub>D</sub> V(D, G) = E<sub>x∼pdata</sub>[log D(x)] + E<sub>z∼pz</sub>[log(1 − D(G(z)))]</p>
  <h4>2.2 Variational Autoencoders (VAEs)</h4>
  <p><strong>Need:</strong><br>
    VAEs enable the learning of meaningful latent representations, which are critical for cross-domain knowledge fusion.
  </p>
  <p><strong>Mathematical Formulation (Evidence Lower Bound - ELBO):</strong></p>
  <p style="text-align: center;">L(θ,ϕ) = − E<sub>qϕ(z|x)</sub>[log pθ(x|z)] + KL(qϕ(z|x) ∥ p(z))</p>
  <p>
    where <em>θ</em> and <em>ϕ</em> denote the decoder and encoder parameters, respectively, and <em>KL</em> is the Kullback-Leibler divergence.
  </p>
  <h4>2.3 Model-Agnostic Meta-Learning (MAML)</h4>
  <p><strong>Need:</strong><br>
    Meta-learning frameworks such as MAML allow the model to quickly adapt to new tasks with minimal data, which is vital for autonomous operation.
  </p>
  <p><strong>Mathematical Formulation:</strong><br>
    For a given task T<sub>i</sub>, the inner-loop update is:
  </p>
  <p style="text-align: center;">θ′ = θ − α ∇<sub>θ</sub> L<sub>T<sub>i</sub></sub>(f<sub>θ</sub>)</p>
  <p>
    followed by a meta-update:
  </p>
  <p style="text-align: center;">θ ← θ − β ∇<sub>θ</sub> &Sigma;<sub>T<sub>i</sub>∼p(T)</sub> L<sub>T<sub>i</sub></sub>(f<sub>θ′</sub>)</p>
  <p>
    where <em>α</em> and <em>β</em> are the inner and meta learning rates, respectively.
  </p>
</div>`
    };

    publicationContent.innerHTML = publicationData[publicationId] || "<p>Publication not found.</p>";
    setActiveTab("tab_publication");
  }

  // data science stories 
function loadStory(storyId) {
  const storyData = {
    "story1": {
      "content": `
      <div style="font-family: 'Georgia', serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px;">
        <h2 style="text-align:center; margin-bottom: 20px;">Whispering with Siri and the Language of Data</h2>
        <p>The genesis of Siri was more than a technological challenge—it was about teaching machines to understand us. At the heart of this transformation was a blend of Natural Language Processing (NLP) techniques and innovative algorithms:</p>
        <ul style="margin: 20px 0; padding-left: 20px;">
          <li><strong>Deep Neural Networks (DNNs):</strong> These models formed the backbone of Siri’s ability to parse and understand spoken language, capturing context and nuances that a mere keyword-based system could never achieve.</li>
          <li><strong>Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks:</strong> By remembering previous parts of a conversation, these algorithms enabled Siri to maintain context, turning disjointed commands into coherent conversations.</li>
          <li><strong>Federated Learning:</strong> This groundbreaking approach allowed us to train models across millions of devices without compromising privacy—ensuring that each interaction refined Siri’s abilities while keeping personal data secure.</li>
        </ul>
        <p>One morning, I tested Siri with my usual trick—asking about the weather in three different ways.</p>
        <p style="margin: 10px 0;">
          ❌ “Hey Siri, what’s the weather today?” (Easy.)<br>
          ❌ “Do I need an umbrella?” (Harder.)<br>
          ✅ “If I leave in an hour, will I get wet?” (Magic.)
        </p>
        <p>That moment? That was the power of data—context, prediction, and personalization—all working together invisibly.</p>
        <h3 style="margin-top: 40px;">Capturing Light, Crafting Memories</h3>
        <p>The iPhone camera is more than a lens; it’s a storyteller, a visionary that sees beyond the obvious. In an era when competitors were chasing megapixel counts, we turned our gaze to the future—computational photography. I vividly recall a late-night session testing a prototype camera in near darkness. The challenge was clear: capture the unseen beauty of a dimly lit cityscape without sacrificing detail.</p>
        <p>Enter <strong>Neural Image Signal Processing (NISP)</strong>—a marvel of real-time AI. This technology merged multiple frames, adjusted exposure, balanced colors, and minimized noise, all in a heartbeat. The result was a photograph that revealed hidden layers of light and shadow, a picture that didn’t just record reality but reimagined it. That night, as I gazed at the vibrant, lifelike image, I realized we were not just taking photos—we were crafting memories.</p>
        <h3 style="margin-top: 40px;">The Unseen Guardian of Trust</h3>
        <p>At the core of Apple’s data revolution lies a steadfast commitment to privacy. While others might have seen data as a commodity, we regarded it as a sacred trust. Every innovation was built on a foundation of on-device processing, encryption, and anonymization. Our approach wasn’t merely technical—it was ethical. We believed that the true power of data should never come at the expense of personal freedom. This ethos of privacy-first innovation isn’t just a policy; it’s the soul of Apple.</p>
        <p>When it comes to capturing memories, Apple’s iPhone camera transcends traditional photography. At its core is a suite of algorithms that turn light into art:</p>
        <ul style="margin: 20px 0; padding-left: 20px;">
          <li><strong>Convolutional Neural Networks (CNNs):</strong> These are the unsung heroes of our computational photography. CNNs analyze each pixel to enhance details, adjust lighting, and detect objects—creating images that are not just pictures, but visual narratives.</li>
          <li><strong>Neural Image Signal Processing (NISP):</strong> By merging multiple frames using advanced deep learning techniques, NISP adjusts exposure and balances colors in real time. This process transforms a simple click into a meticulously crafted masterpiece.</li>
          <li><strong>Clustering Algorithms:</strong> Techniques like k-means help segment images, identify patterns, and isolate features, ensuring that every photo you take is as crisp and vibrant as the memory it captures.</li>
        </ul>
        <h3 style="margin-top: 40px;">The Ethical Core—Data Privacy and Trust</h3>
        <p>Amid the whirlwind of innovation, one principle remained unwavering: privacy. At Apple, data was treated not as a commodity, but as a trust to be safeguarded with cutting-edge encryption and ethical algorithms:</p>
        <ul style="margin: 20px 0; padding-left: 20px;">
          <li><strong>On-Device Machine Learning:</strong> By processing data locally, our models ensured that your personal information stayed secure on your device, never venturing into the cloud.</li>
          <li><strong>Anonymization Techniques:</strong> Before any data was used for training, it was rigorously anonymized. This ensured that while our algorithms learned from patterns, they never compromised individual privacy.</li>
          <li><strong>Differential Privacy:</strong> This statistical technique allowed us to analyze large datasets while obscuring the details of any individual entry, balancing the need for data insights with the sanctity of personal information.</li>
        </ul>
      </div>
      `
    },
    "story2": {
      "content": `
      <div style="font-family: 'Georgia', serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px;">
        <h2 style="text-align:center; margin-bottom: 20px;">Google</h2>
        <p>Let me tell you a story—one that starts in the late 90s, when the internet was a sprawling mess of information. In 1998, Google was more of a hope than a reality.</p>
        <p>But then, in the middle of all this chaos, PageRank emerged. PageRank changed everything. It didn’t just look at how many times a keyword appeared on a page. It looked at how pages were connected to each other, like a web of knowledge.</p>
        <p>At the time, the biggest challenge wasn’t just building an algorithm—it was handling the volume. Google had data, but how could it possibly scale to accommodate all of it? Enter <strong>BigTable</strong>—a distributed storage system we built. This wasn’t your regular database. BigTable was engineered to scale to handle massive amounts of unstructured data (think millions of pages, images, user interactions, etc.).</p>
        <p>That’s when we began experimenting with something called <strong>MapReduce</strong>. This was the first step towards parallel computing, where we could split big data tasks across multiple machines and process them simultaneously. Imagine trying to analyze billions of data points on user search behavior—MapReduce allowed us to do that at scale.</p>
        <p>Now, let’s fast-forward to the 2010s. By this time, we were in the midst of a machine learning renaissance. Google wasn’t just a search engine anymore. We had Gmail, Google Maps, YouTube, and Google Photos—each one generating massive amounts of data. And we had to figure out how to not just process it, but use it to make better predictions and build smarter products.</p>
        <p>I remember the early days of Google Translate—I was there. It was a challenge, and it was exciting. Statistical machine translation didn’t work as well as we hoped. Languages don’t work in predictable ways, and simply matching words from one language to another wasn’t enough. So we turned to neural networks—this was the beginning of deep learning, the algorithm that would change everything.</p>
        <p>The breakthrough moment came when we began applying deep learning to Google Photos. It was using computer vision to understand the context of images. You could search for “beach” and instantly get all the photos you took during summer vacations, even though you never labeled them. This was made possible by algorithms like <strong>Convolutional Neural Networks (CNNs)</strong>, which are designed to work on image data.</p>
        <p>Training these models wasn’t a walk in the park. Deep learning models are notoriously computationally expensive. It took months of fine-tuning, sometimes hundreds of GPUs, and at times, we even had to build custom chips (yes, TPUs—Tensor Processing Units) just to speed up training. The real magic of Google wasn’t just that we had access to all this data; it was that we built tools and algorithms that allowed us to make sense of it, fast.</p>
      </div>
      `
    },
    "story3": {
      "content": `
      <div style="font-family: 'Georgia', serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px;">
        <h2 style="text-align:center; margin-bottom: 20px;">OPEN AI</h2>
        <h3>How GPT-3 Became a Revolutionary Language Model</h3>
        <p>In 2020, when GPT-3 was launched, it wasn’t just another language model—it was a leap forward in conversational AI. We trained GPT-3 with billions of parameters, allowing it to generate text that felt almost indistinguishable from human speech. The real breakthrough came when GPT-3 could handle nuanced conversations, understand context, and even produce creative content like poetry and code. But what really set it apart was its few-shot learning ability, allowing it to understand tasks with minimal input.</p>
        <h3 style="margin-top: 40px;">DALL·E and the Power of AI-Driven Art</h3>
        <p>When DALL·E was first unveiled, we knew it was going to change how we approached creativity. The challenge wasn’t just generating images, but creating new visual concepts based on text descriptions. Using a unique architecture, DALL·E could transform abstract text into vivid images, showing the world how AI could assist in artistic expression. From surrealist interpretations to simple designs, DALL·E opened a new door for artists, designers, and storytellers.</p>
        <h3 style="margin-top: 40px;">OpenAI Codex—Empowering Developers with AI</h3>
        <p>In 2021, OpenAI Codex became a game-changer for developers. Powered by the same GPT-3 technology, Codex could understand both human language and programming languages, turning natural language commands into code. The ability for Codex to generate Python, JavaScript, and even SQL with ease was a turning point in software development. Developers could now spend more time focusing on problem-solving and less on syntax, making programming accessible to a wider range of people.</p>
        <h3 style="margin-top: 40px;">CLIP—Bridging the Gap Between Vision and Language</h3>
        <p>CLIP (Contrastive Language-Image Pretraining) revolutionized how AI understood images and text together. Rather than being trained separately, CLIP was designed to understand both text and visual data simultaneously, making it one of the first truly multimodal models. The magic was in its ability to match images with textual descriptions, making it possible to search for images with natural language and generate captions for visuals. CLIP is now at the heart of many creative and practical applications, from content moderation to visual search engines.</p>
        <h3 style="margin-top: 40px;">The Dilemma of Bias and Fairness in AI</h3>
        <p>While we built cutting-edge models, we were also acutely aware of the ethical challenges posed by AI. OpenAI dedicated significant resources to addressing bias in our models. By constantly iterating on our algorithms and refining the training data, we sought to reduce harmful bias and promote fairness in every product we released. For us, AI wasn’t just about building better models; it was about building responsible models.</p>
        <h3 style="margin-top: 40px;">GPT-4—The Next Generation of AI Understanding</h3>
        <p>GPT-4 was more than just a larger model; it was a fundamentally more capable one. With a better grasp of subtle nuances, GPT-4’s ability to understand complex instructions, reason logically, and generate human-like responses made it a tool that pushed boundaries in fields like healthcare, law, and content creation. What was truly impressive was how GPT-4 managed to take even the most sophisticated queries and provide responses that were grounded in factual accuracy and context.</p>
      </div>
      `
    }
  };

  if (storyData[storyId]) {
    // Set the inner HTML for the story content
    storyContent.innerHTML = storyData[storyId].content;
    
    // Reset the overall page background
    document.body.style.backgroundImage = "";
    document.body.style.backgroundColor = "#f0f0f0";

    // Choose the appropriate background image URL for each story
    let bgImageUrl = "";
    if (storyId === "story1") {
      bgImageUrl = "https://raw.githubusercontent.com/bolleddu15/pbblog/main/a.jpg";
    } else if (storyId === "story2") {
      bgImageUrl = "https://raw.githubusercontent.com/bolleddu15/pbblog/main/g.png";
    } else if (storyId === "story3") {
      bgImageUrl = "https://raw.githubusercontent.com/bolleddu15/pbblog/main/o.jpg";
    }
    
    // Apply the background image to the story container with a white overlay
    // The linear-gradient creates a white layer with 75% opacity over the image,
    // making the image appear lightly visible.
    storyContent.style.backgroundImage = `linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.75)), url('${bgImageUrl}')`;
    storyContent.style.backgroundSize = "cover";
    storyContent.style.backgroundRepeat = "no-repeat";
    storyContent.style.backgroundPosition = "center";
    // Remove any solid background color so the image shows through
    storyContent.style.backgroundColor = "transparent";
    
    // Other styling for the story container
    storyContent.style.padding = "20px";
    storyContent.style.borderRadius = "10px";
    storyContent.style.color = "black";
  } else {
    storyContent.innerHTML = "<p>Story not found.</p>";
  }
  setActiveTab("tab_story");
}


  // Listen for clicks on main navigation links
  document.querySelectorAll(".tab-link").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      window.location.hash = link.getAttribute("href");
    });
  });

  // Listen for clicks on project links
  document.querySelectorAll(".project-link").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      window.location.hash = link.getAttribute("href");
    });
  });

  // Listen for clicks on publication links
  document.querySelectorAll(".publication-link").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      window.location.hash = link.getAttribute("href");
    });
  });

  // Listen for clicks on story links
  document.querySelectorAll(".story-link").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      window.location.hash = link.getAttribute("href");
    });
  });

  // Function to handle hash-based routing
  function handleRouting() {
    const hash = window.location.hash || "#/";
    if (hash === "#/" || hash === "#") {
      setActiveTab("tab_1");
      return;
    }
    if (hash.startsWith("#/projects")) {
      if (hash.includes("predictive-maintenance")) {
        loadProject("project1");
      } else if (hash.includes("customer-segmentation")) {
        loadProject("project2");
      } else if (hash.includes("academic-chatbot")) {
        loadProject("project3");
      } else if (hash.includes("healthcare-classification")) {
        loadProject("project4");
      } else {
        setActiveTab("tab_2");
      }
    } else if (hash.startsWith("#/publications")) {
      if (hash.includes("An%20Analysis%20of%20Clustering%20Techniques")) {
        loadPublication("pub1");
      } else if (hash.includes("Hadoop")) {
        loadPublication("pub2");
      } else if (hash.includes("Neuro-Symbolic")) {
        loadPublication("pub3");
      } else if (hash.includes("Self-Sustaining")) {
        loadPublication("pub4");
      } else {
        setActiveTab("tab_3");
      }
    } else if (hash.startsWith("#/stories")) {
      if (hash.includes("story1")) {
        loadStory("story1");
      } else if (hash.includes("story2")) {
        loadStory("story2");
      } else if (hash.includes("story3")) {
        loadStory("story3");
      } else {
        setActiveTab("tab_4");
      }
    } else {
      setActiveTab("tab_1");
    }
  }

  window.addEventListener("hashchange", handleRouting);
  handleRouting();
});
