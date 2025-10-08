# Contextual AI Parser + Weaviate Integration

This directory contains a recipe for integrating [Contextual AI Parser](https://docs.contextual.ai/api-reference/parse/parse-file) with [Weaviate](https://weaviate.io/) to build powerful RAG (Retrieval-Augmented Generation) applications over PDF documents.

## Overview

The integration demonstrates how to:

1. **Parse different document types** using Contextual AI's specialized parsing capabilities
2. **Extract structured content** with document hierarchy preservation and table intelligence
3. **Process and chunk** content with document-type-specific strategies
4. **Store embeddings** in Weaviate with rich metadata for precise filtering
5. **Perform multi-modal RAG** across different document types simultaneously

## Files

- `rag_over_pdfs_contextual_weaviate.ipynb` - Complete notebook demonstrating the integration

## Key Features

### Contextual AI Parser Advantages
- **Cloud-based processing**: No local compute requirements
- **High-quality extraction**: Advanced OCR and structure recognition
- **Document hierarchy**: Preserves document structure for better context
- **Table intelligence**: Smart table extraction and splitting for large documents
- **Multiple formats**: Markdown and JSON outputs
- **Production-ready**: Scalable cloud service

### Weaviate Integration Benefits
- **Vector search**: Semantic similarity search across parsed content
- **Generative AI**: Built-in RAG with OpenAI integration
- **Rich metadata**: Support for filtering and precise retrieval
- **Scalability**: From embedded development to cloud production

## Documents Used

This recipe demonstrates parsing two distinct document types:

1. **Research Paper**: "Attention is All You Need" - A seminal transformer architecture paper
   - Demonstrates document hierarchy preservation
   - Shows figure caption extraction
   - Parses first 6 pages for efficiency

2. **Table-Rich Document**: OmniDocBench dataset documentation
   - Demonstrates advanced table extraction capabilities
   - Shows table splitting with header propagation
   - Handles large, complex tables

## Requirements

- Contextual AI API key
- OpenAI API key
- Python environment with required packages

## Usage

1. Set up your API keys (Contextual AI and OpenAI)
2. Run the notebook cells in sequence
3. The notebook will:
   - Download and parse two distinct document types
   - Process content with document-type-specific strategies
   - Store embeddings with rich metadata in Weaviate
   - Demonstrate multi-modal RAG queries

## Example Queries

The notebook includes examples for:
- Transformer architecture and attention mechanisms
- Dataset benchmarks and performance metrics
- Cross-document analysis and comparison

## Key Differentiators

### From Other Parsers:
- **Hierarchical context**: Parent-child relationships preserved in chunks
- **Table intelligence**: Large tables automatically split with context preservation
- **Document type awareness**: Different parsing strategies for different content types
- **Rich metadata**: Document structure information enhances RAG quality

### From Local Parsing:
- **No GPU required**: Cloud-based processing
- **Enterprise-grade**: Production-ready service
- **Advanced features**: Table splitting, hierarchy preservation, multiple output formats

## Next Steps

Consider extending this integration with:
- Document-level metadata for better source attribution
- Hybrid search combining keyword and semantic search
- Different chunking strategies for each document type
- Advanced RAG frameworks like [DSPy](https://weaviate.io/developers/integrations/llm-frameworks/dspy) or [LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/weaviate/)
- [Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag) for complex multi-step queries

## Links

- [Contextual AI Parser Documentation](https://docs.contextual.ai/api-reference/parse/parse-file)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Generative Search](https://weaviate.io/developers/weaviate/search/generative)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)