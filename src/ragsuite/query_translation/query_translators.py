import logging
from typing import Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ragsuite.core.config import load_conf
from ragsuite.core.ports import LLMClient
from ragsuite.core.types import QueryList, TranslationContext, QueryStr

logger: logging.Logger = logging.getLogger(__name__)


# Since all query translation methods share the same steps, I've implemented
# a general solution, which only needs a prompt. It is used via composition
# in the specific translators.
class _QueryTranslatorImpl:
    def __init__(
        self,
        chat_model: LLMClient,
        prompt_templ: PromptTemplate,
    ):
        self.chat_model = chat_model
        self.prompt_templ = prompt_templ

    def run(self, ctx_dict: Dict) -> QueryList:
        logger.debug("Running _QueryTranslatorImpl")
        ctx_dict = ctx_dict or {}
        chain = (
            self.prompt_templ
            | self.chat_model
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        llm_response = chain.invoke(ctx_dict)
        # Remove dupliates while preserving order.
        seen = set()
        for s in llm_response:
            s_norm = s.strip()
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
        return QueryList(
            original_query=QueryStr(ctx_dict.get("query")),
            queries=list(seen),
        )


class BaseTranslator:
    def __init__(self, chat_model: LLMClient, prompt_templ: PromptTemplate):
        self.chat_model = chat_model
        self.prompt_templ = prompt_templ
        self._impl: Optional[_QueryTranslatorImpl] = None

    def initialize_impl(self):
        if self._impl is not None:
            return
        self._impl = _QueryTranslatorImpl(
            chat_model=self.chat_model, prompt_templ=self.prompt_templ
        )

    def translate(
        self,
        ctx: TranslationContext
    ) -> QueryList:
        self.initialize_impl()
        ctx_dict = ctx.to_dict()
        return self._impl.run(ctx_dict)


class MultiQueryTranslator(BaseTranslator):
    def __init__(self, chat_model: LLMClient):
        logger.debug("Starting MultiQueryTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templ_lst.multi_query_rag_prompt.input_variables,
                template=conf.prompt_templ_lst.multi_query_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class HyDETranslator(BaseTranslator):
    def __init__(self, chat_model: LLMClient):
        logger.debug("Starting HyDETranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templ_lst.hyde_rag_prompt.input_variables,
                template=conf.prompt_templ_lst.hyde_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class DecompositionTranslator(BaseTranslator):
    def __init__(self, chat_model: LLMClient):
        logger.debug("Starting DecompositionTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templ_lst.decomposition_rag_prompt.input_variables,
                template=conf.prompt_templ_lst.decomposition_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class StepBackTranslator(BaseTranslator):
    def __init__(self, chat_model: LLMClient):
        logger.debug("Starting StepBackTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templ_lst.step_back_rag_prompt.input_variables,
                template=conf.prompt_templ_lst.step_back_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class IdentityTranslator(BaseTranslator):
    def __init__(self, chat_model: Optional[LLMClient] = None):
        logger.debug("Starting IdentityTranslator initialization")
        # Identity translator doesn't need a chat model.

    def translate(
        self,
        ctx: TranslationContext
    ) -> QueryList:
        return QueryList(
            original_query=ctx.query, queries=[ctx.query]
        )
