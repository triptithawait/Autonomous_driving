# Viva Questions & Answers: Smart Navigation System

**Q1: Why use Fuzzy Logic instead of simple IF-ELSE thresholds?**
*A: Fuzzy logic handles the "vagueness" of real-world data. A road isn't just "narrow" or "not narrow"; it can be "somewhat narrow." Fuzzy logic allows for smooth transitions and a more human-like decision-making process, providing a suitability percentage rather than a binary YES/NO.*

**Q2: What is the role of CNN in this project?**
*A: The CNN (Convolutional Neural Network) acts as the perceptual module. It processes raw visual data (images/video frames) to identify environmental features—in this case, classifying the road type to estimate its width.*

**Q3: How does Dijkstra's algorithm incorporate the suitability score?**
*A: We modify the weight of each edge in the graph. Instead of just using distance (L), we use L / (Suitability Score). If a road is highly unsuitable (score near 0), its effective "search cost" becomes near infinite, causing the algorithm to bypass it in favor of a wider, more suitable route.*

**Q4: Which membership functions were used in the Fuzzy Logic system?**
*A: We used triangular membership functions (trimf) for simplicity and efficiency. They are computationally inexpensive and effectively represent the ranges for narrow, medium, and wide roads.*

**Q5: What are the limitations of using MobileNet for road width estimation?**
*A: While MobileNet is fast and lightweight, it may lack the depth to differentiate very similar road textures in poor lighting. For higher accuracy, one might need a segmentation model like U-Net or SegNet to measure actual pixels of the road surface.*

**Q6: How can this system be scaled for a real city?**
*A: By integrating OpenStreetMap (OSM) data. OSM provides "tags" for many roads (like lane counts or width). The system could use OSM as a base layer and the CNN as a real-time verification/update layer.*

**Q7: What is the benefit of using "Linguistic Variables" in Fuzzy Logic?**
*A: It makes the system interpretable. Rules like "IF Road is Narrow AND Vehicle is Large THEN Suitability is Very Low" are easy for engineers to define and for users to understand compared to complex mathematical weights.*
