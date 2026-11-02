# BIFS 614 Expert System

**2026 Fall Project for BIOT 670I – Biotechnology Capstone: Bioinformatics**  
**Team Members:** Sara Drennan, George Hill, Michael Jersey, and Edward Matovu

---

## Project Overview

The BIFS 614 Expert System is a capstone project focused on the development of a Large Language Model (LLM)-based expert system for the course BIFS 614 – Data Structures and Algorithms.

The goal of the project is to develop an LLM expert for the BIFS 614 course capable of operating in different learning environments.

The original project objectives included two distinct modes:

- **TUTOR Mode** – An online interactive tutor for students to use as a learning resource to ask questions and receive answers based on relevant course content.
- **PROCTOR Mode** – A restricted mode designed for online examination scenarios, with appropriate behavioral guardrails and restrictions to prevent the system from providing unauthorized assistance.

Throughout development, the project expanded to include multiple versions of Tutor Mode, providing students with different ways to interact with and learn from the course materials.

This repository contains the course-content resources, preprocessing scripts, knowledge base, and CourseAgent Python scripts developed throughout the project.

---

## Project Development and Status

The BIFS 614 Expert System has progressed through several stages of development, including:

- Collection, conversion, cleaning, and validation of the BIFS 614 course materials to create a course-specific knowledge base.
- Selection and testing of the Microsoft Phi-3 Mini language model.
- Implementation of Retrieval-Augmented Generation (RAG) using text chunking, embeddings, and FAISS vector search.
- Development of `build_index.py` to process the course knowledge base and generate the files needed for content retrieval.
- Development and testing of multiple CourseAgent Python scripts, with improvements to retrieval settings, system prompts, and guardrails to help ensure responses remain grounded in the provided course materials.
- Development of four Tutor Mode versions, each providing different learning and interaction options.
- Integration of Streamlit to provide a web-based interface for students to access and interact with the different Tutor Modes.

---

## Accessing the BIFS 614 Expert System

The BIFS 614 Expert System includes four different Tutor Mode versions that can be accessed through Streamlit.

**To use the Expert System, click on one of the Streamlit links below.** Once the application has loaded, students can ask questions related to the BIFS 614 course materials or interact with the available learning features.

### Available Tutor Modes

**Version 7 – Question and Answer Tutor**

Students can ask questions related to the BIFS 614 course content and receive responses based on the course knowledge base.

**Streamlit Link:** https://bifs614expertprojectfall2026-courseagentv7.streamlit.app/

**Version 8 – Interactive Tutor**

An interactive tutor mode that allows students to explore course topics, review concepts, and request additional explanations.

**Streamlit Link:** https://bifs614expertprojectfall2026-courseagentv8.streamlit.app/

**Version 9 – Practice Tutor**

A practice-based tutor mode where students receive more in-depth responses and explanations, and answer course-related questions with feedback on their responses.

**Streamlit Link:** https://bifs614expertprojectfall2026-courseagentv9.streamlit.app/

**Version 10 – Combined Interactive Tutor and Practice Questions**

Combines the interactive learning features of Tutor Modes in Version 7, 8, and 9 including an option for practice questions, allowing students to review course concepts and test their understanding.

**Streamlit Link:** https://bifs614expertprojectfall2026-courseagentv10.streamlit.app/

---

## Running the Project Locally (Optional)

The project can also be run locally using Python.

1. Clone or download the GitHub repository.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. If the FAISS index files need to be generated or updated, run:

   ```bash
   python scripts/build_index.py
   ```

4. Run the desired CourseAgent Python script or its corresponding Streamlit application.

The repository contains the Python scripts, course knowledge base, and supporting files used throughout our development.

---

## Project Technologies

The primary tools and technologies utilized throughout the development of the BIFS 614 Expert System include:

- Python
- Microsoft Phi-3 Mini 4K Instruct
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS Vector Search
- Streamlit
- PyCharm
- GitHub
- UMGC Hosted Lab

---
