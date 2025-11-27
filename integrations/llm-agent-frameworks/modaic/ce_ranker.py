from dotenv import load_dotenv
import os
import asyncio

import dspy
from modaic import PrecompiledAgent, PrecompiledConfig
import weaviate

load_dotenv()

class RelevanceAssessment(dspy.Signature):
    """Assess the relevance of a document to a query."""
    query: str = dspy.InputField()
    document: str = dspy.InputField()
    relevance_score: bool = dspy.OutputField()


class CERankerConfig(PrecompiledConfig):
    collection_name: str
    return_properties: list[str]
    k: int
    lm: str = "openai/gpt-4.1-mini"


class CERankerAgent(PrecompiledAgent):
    config: CERankerConfig

    def __init__(self, config: CERankerConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        lm = dspy.LM(self.config.lm)
        dspy.configure(lm=lm)
        
        self._connect_to_weaviate()
        self.k = config.k
        self.reranker = dspy.ChainOfThought(RelevanceAssessment)

    def _connect_to_weaviate(self):
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )
        self.collection = self.weaviate_client.collections.get(
            self.config.collection_name
        )

    async def _score_document(self, query: str, document: str) -> tuple[str, bool]:
        result = await self.reranker.acall(query=query, document=document)
        return (document, result.relevance_score)

    async def __acall__(self, query: str, k: int | None = None) -> list[str]:
        if k is None:
            k = self.k

        response = self.collection.query.hybrid(query=query, limit=k)
        documents = [o.properties["content"] for o in response.objects]

        scored_results = await asyncio.gather(
            *[self._score_document(query, doc) for doc in documents]
        )

        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_results[:k]]

    def __call__(self, query: str, k: int | None = None) -> list[str]:
        return asyncio.run(self.__acall__(query, k))


if __name__ == "__main__":
    config = CERankerConfig(
        collection_name="IRPapersText_Default",
        return_properties=["content"],
        k=5
    )
    agent = CERankerAgent(config)
    print(agent(query="What is HyDE?"))
    agent.push_to_hub(
        "connor/CrossEncoderRanker",
        with_code=True,
        commit_message="Fix init to accept kwargs"
    )