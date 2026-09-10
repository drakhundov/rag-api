import logging
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ragsuite.core.ports import Retriever, LLMClient
from ragsuite.routing import HeuristicRouter
from ragsuite.core.types import RRFConfig, QueryStr, QueryList, TranslationContext
from ragsuite.utilities import cli

logger: logging.Logger = logging.getLogger(__name__)

#TODO possibly implement lemmatization


class RAGEngine:
    def __init__(
        self, doc_retriever: Retriever, chat_model: LLMClient, sys_prompt_template: PromptTemplate,
        rrf_conf: RRFConfig | None
    ):
        self.doc_retriever = doc_retriever
        self.chat_model = chat_model
        self.sys_prompt_template = sys_prompt_template
        if rrf_conf is not None:
            self.rrf_conf = rrf_conf
        else:
            self.rrf_conf = RRFConfig(top_k=5, k_rrf=60)

    def generate_answer(self, query: QueryStr, top_k: int = 4) -> str:
        logger.debug(f"Generating answer for query: {query}")
        router = HeuristicRouter(
            ctx=TranslationContext(query=query, quantity=top_k, max_tokens=256),
            chat_model=self.chat_model
        )
        router.build_route()
        qlist: QueryList = router.run_route()
        docs: List[List[Document]] = []
        for q in qlist:
            docs.append(self.doc_retriever.retrieve(q, top_k=top_k))
        # Weed out the most relevant documents using Reciprocal Rank Fusion.
        ranked_docs: List[Document] = self.perform_rrf(docs)
        return self.chat_model.generate(self.sys_prompt_template, query, ranked_docs)

    @cli.with_temp_message(message="Performing reciprocal rank fusion...")
    def perform_rrf(self, docs: List[List[Document]]) -> List[Document]:
        top_k = self.rrf_conf.top_k
        k_rrf = self.rrf_conf.k_rrf
        logger.debug("Performing reciprocal rank fusion")
        scores: Dict[str, float] = {}
        first_seen: Dict[str, Document] = {}
        for ranking in docs:
            for r, doc in enumerate(ranking, start=1):
                doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id") or doc.page_content[:80]
                if doc_id not in first_seen:
                    first_seen[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + r)
        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [first_seen[doc_id] for doc_id, _ in fused][:top_k]
