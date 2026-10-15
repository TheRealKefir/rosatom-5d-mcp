# server.py
from prompts.registry import PromptRegistry

PROMPTS = PromptRegistry("prompts")

# @mcp.tool()
def calculate_print_parameters() -> str:
    pass
    # msgs = RAGBuilder(PROMPTS).build(user_query=..., material=rag_hit)
    # raw = llm.chat(msgs, temperature=PROMPTS.analyze_ctl_geometry.meta.temperature)
    # log.info("used prompt: %s@%s", PROMPTS.param_calculator.meta.name,
    #                             PROMPTS.param_calculator.meta.version)
    # return validate_and_fix_json(raw)