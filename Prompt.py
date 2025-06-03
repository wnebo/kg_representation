generate_entity_types = '''You are given a list of natural language questions (queries) that will be used to guide knowledge graph construction. Your task is to analyze these queries and summarize the types of entities and the types of relationships that will be needed to answer them.

Please follow these instructions:

1. Read through the entire query list.
2. Identify the kinds of entities that are likely to appear in the answers (e.g., "person", "organization", "product", "country", "date", etc.)
3. Identify the types of relationships implied by the questions (e.g., "founded_by", "located_in", "ceo_of", "produces", etc.)
4. Try to use general but informative type names. Avoid repeating the same type under different names unless necessary.
5. Output your response in the following JSON format:

```json
{
  "entity_types": ["..."],
  "relation_types": ["..."]
}
'''

financial_entity_types = [
    # 企业与组织
    "company",
    "subsidiary",
    "parent_company",
    "startup",
    "financial_institution",
    "investment_bank",
    "commercial_bank",
    "central_bank",
    "hedge_fund",
    "private_equity_firm",
    "venture_capital_firm",
    "insurance_company",
    "rating_agency",
    "regulatory_agency",
    "exchange",
    "government_agency",
    "organization",
    "legal_entity",

    # 人与角色
    "person",
    "founder",
    "investor",
    "executive",
    "ceo",
    "cfo",
    "analyst",
    "trader",
    "board_member",
    "regulator",

    # 金融工具与产品
    "financial_product",
    "stock",
    "share",
    "bond",
    "corporate_bond",
    "sovereign_bond",
    "municipal_bond",
    "derivative",
    "option",
    "future",
    "etf",
    "mutual_fund",
    "index",
    "loan",
    "mortgage",
    "structured_product",
    "cds",  # credit default swap
    "currency",
    "commodity",
    "digital_asset",
    "crypto_token",

    # 市场与经济指标
    "market",
    "sector",
    "industry",
    "exchange_rate",
    "interest_rate",
    "inflation_rate",
    "economic_indicator",
    "gdp",
    "cpi",
    "ppi",
    "unemployment_rate",
    "credit_rating",
    "risk_factor",

    # 行为与事件
    "merger_acquisition",
    "investment",
    "ipo",
    "bankruptcy",
    "scandal",
    "policy_decision",
    "lawsuit",
    "financial_crisis",
    "regulatory_action",
    "audit",

    # 时间与数值类
    "date",
    "amount",
    "percentage",
    "duration",
    "valuation",
    "stock_price",
    "market_cap"
]

async def _process_document(
    self, text: str, prompt_variables: dict[str, str]
) -> str:
    history_messages = []
    results = ""

    # 初始提问
    user_prompt = self._extraction_prompt.format(**{
        **prompt_variables,
        self._input_text_key: text,
    })
    history_messages.append(f"User: {user_prompt}")
    response = await self._model.achat("\n".join(history_messages))
    content = response.output.content or ""
    results += content
    history_messages.append(f"Assistant: {content}")

    # 多轮抽取
    if self._max_gleanings > 0:
        for i in range(self._max_gleanings):
            history_messages.append(f"User: {CONTINUE_PROMPT}")
            response = await self._model.achat("\n".join(history_messages))
            content = response.output.content or ""
            results += content
            history_messages.append(f"Assistant: {content}")

            if i >= self._max_gleanings - 1:
                break

            history_messages.append(f"User: {LOOP_PROMPT}")
            response = await self._model.achat("\n".join(history_messages))
            loop_check = (response.output.content or "").strip()
            history_messages.append(f"Assistant: {loop_check}")

            if loop_check != "Y":
                break

    return results
