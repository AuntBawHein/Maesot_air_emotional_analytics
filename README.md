# Maesot_air_emotional_analytics
An NLP or Data Engineering pipeline quantifying Mae Sot's PM2.5 smoke crisis. Translates Thai/English survey stories into sentiment or subjectivity scores, mapping physical air pollution directly to community well-being.

### **1. The Research Problem**
*   **The Issue:** While technical sensors provide raw **Mae Sot air quality** numbers, there is no organized data documenting how the community actually experiences these conditions.

*   **The Missing Link:** I observed a "Safety Gap" where medical professionals wear masks 100% of the time, yet many in the public do not, proving that technical data alone does not always lead to protective behavior.

*   **The Solution:** I am building a "Story-to-Math" process using AI to turn personal human experiences into scientific information to map the total impact of the smoke crisis on our community.

  ### **2. Project Aims (The Primary Goals)**
*   **Harmonized Data:** I aim to construct a data engineering system that ingests multilingual community feedback (Thai and English) and harmonizes it into a single English framework for analysis.

*   **Quantified Well-being:** I will use **Linguistic AI** (TextBlob) to quantify the relationship between environmental hazards and community well-being using sentiment and subjectivity scores.


### **3. Project Scope, Boundaries & Limitations**

*   **Geographic Boundary:** I focused specifically on the **Mae Sot and Tak border area** to ensure the data is locally relevant.

*   **Data Volume & Structure:** The project analyzed **77 real-world responses** (48 Thai and 29 English). To help the AI detect larger patterns, I integrated **73 synthetic "dummy" rows** to reach a total of **150 responses**.

*   **Scientific Honesty:** I have documented the use of synthetic data as a project limitation. I also track the "Survey Language" for every row to identify potential cultural or demographic biases between linguistic groups.

***

### **Project Limitations**

I have documented the use of 73 synthetic "dummy" rows as a limitation to be scientifically honest. I also track the "Survey Language" for every row to identify potential cultural or demographic biases between linguistic groups.


### **The Tech Stack**

I built this system using Python (Core), Pandas (Data Cleaning), Matplotlib and Seaborn (Visualization), and a dual AI layer consisting of Deep-Translator (Cross-lingual Harmonization) and TextBlob (Sentiment Scoring).

*   **TextBlob:** This is the NLP (Natural Language Processing) tool I use to quantify the emotional content and the subjectivity of responses.
*   **Deep-Translator:** This is the machine translation tool I use to turn local Thai responses into English so the AI can process every voice together.
*   **Pandas:** This is the Data Organizer I use to arrange all 150 community responses into a structured table for cleaning and filtering.
*   **Seaborn:** This is the Advanced Visualizer I use to create sophisticated statistical charts like the Human Impact scatter plot.
*   **Matplotlib:** This is the Chart Designer I use to create clear graphs that illustrate the impact of the smoke on our daily lives.
*   **Numpy:** This is the Mathematician I use to perform numerical calculations and find connections between different survey answers.

***


## 4. Objectives (The "How-To" Steps)

### **Step 1: Gathering Evidence & Designing the Survey**

1.  **Real-World Evidence Base:** I collected health guidelines from the **WHO** and the **Thai Ministry of Public Health**, while also researching environmental studies on **Google Scholar** to build a foundation of facts for the **Mae Sot air quality** project.

2.  **The Expert Baseline:** I used advice from **Burmese and Thai doctors** to identify local "Impact Markers"—such as asthma spikes and mountain inversions—to establish a baseline for the **Mae Sot air AQI series**.

3.  **The Survey Design:** I built a **20-question Google Form** using a mix of simple choices and open-ended text boxes. This provides my AI with **information-rich data** (detailed personal stories) to turn personal experiences into measurable scientific information.

***


***

### **Step 2: Connecting Different Languages**

*   **Making the Data Match:** I use a process that automatically finds and translates different languages. I take the raw responses written in Thai and English and turn them all into one single language (English) so I can analyze my **77 real community stories** together as one unified dataset.

*   **Using the Translator:** To make sure my code stays stable and fast, I use the **deep-translator** library and the **GoogleTranslator** module. I use this tool to convert Thai responses into English so my program can process all the data at once without technical errors or network timeouts.

### **Step 3: Sentiment Quantification (Facts vs. Feelings)**

*   **Contextual Extraction:** Once the data is unified, the NLP (Natural Language Processing) engine isolates specific **"Impact Markers"** like sore eyes, N95 mask usage, or housework burdens caused by ash.

*   **Subjectivity Scoring:** I use the **TextBlob** library to calculate a **Subjectivity Score** (0.0 to 1.0) to mathematically distinguish between an objective **Fact** and a personal human **Feeling**. The AI detects "intensity" words—like *very* or *terrible*—to move the score toward 1.0, effectively turning personal stories into measurable scientific data.

***

### **4. Math and Connections**
*   **Expert Check:** I will cross-reference the community's emotional scores with medical criteria to see if public stress levels align with actual air hazards.

*   **Statistical Patterns:** I will use math to trace connections between factors like location or mask-wearing frequency and the overall "Human Impact Index".

### **5. The "Human Impact" Dashboard**
*   **The Main Impact Chart:** I will create a visual comparison showing how the community's mood changes as pollution metrics go up or down.

*   **The Problem Map:** I will build charts to identify the primary daily burdens causing the most struggle for residents. To remain scientifically honest, I will explicitly flag when percentages are calculated from the **150-row master (which includes 73 synthetic dummy rows)** versus the **77 real human responses**.

***



---

## 5. Success Metrics

*   **Technical:** I will successfully build an automated system that reads survey spreadsheets and extracts clear, mathematical facts from human stories.

*   **Community:** I will create a "Human Impact Dashboard" to provide local leaders with objective proof of Mae Sot’s health and safety needs.

*   **Educational:** I will master the skills of combining different datasets, using AI to bridge language barriers, and learning how to explain my project's data limitations with scientific honesty.

***

### **1. Loading Python Libraries**

In **Step 1**, I am preparing my software environment by loading the specific Python libraries required for my data science analysis.

*   I am using **Pandas** to organize all community survey responses into a structured table for cleaning and analysis.

*   I am importing **NumPy** to act as a mathematician for performing numerical calculations.

*   I am loading **TextBlob**, which is the AI logic I use to translate Thai responses and measure the emotional intensity of the community.

*   I am adding **Matplotlib** and **Seaborn** to create clear, advanced charts that illustrate the impact of the smoke on people's daily lives.

***

# I import the pandas library and name it 'pd' to help me organize my 77
# survey responses into structured data tables
import pandas as pd

# I import the numpy library as 'np' so I can calculate mathematical
# averages and find patterns in my data
import numpy as np

# I import the matplotlib library as 'plt' to create the bar charts and pie
# charts I need for my final dashboard
import matplotlib.pyplot as plt

# I import the TextBlob tool to act as the AI brain that will translate
#  Thai responses and measure the emotional intensity of the community
from textblob import TextBlob

# I print this message to confirm that my coding workspace is properly set
# up and ready for research
print("Toolkit loaded successfully.")

