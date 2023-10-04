from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# baseline template(zero-shot)
baseline_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}
Event1: {event1}
Event2: {event2}
Your answer: 
"""
baseline_response_schemas = [
    ResponseSchema(
        name="Answer",
        description=("The answer to the question.")
    ),
]
baseline_output_parser = StructuredOutputParser.from_response_schemas(baseline_response_schemas)
baseline_format_instructions = baseline_output_parser.get_format_instructions()
baseline_prompt = PromptTemplate(
    template=baseline_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": baseline_format_instructions},
)
# zero-shot cot template
zero_shot_cot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}
Event1: {event1}
Event2: {event2}
Let's think step by step.
Your reasoning:
Your answer:
"""
zero_shot_cot_response_schemas = [
    ResponseSchema(
        name="Reasoning",
        description="The reasoning for the answer. Your thinking process."
    ),
    ResponseSchema(
        name="Answer",
        description="The final answer to the question, given the reasoning."
    ),
]
zero_shot_cot_output_parser = StructuredOutputParser.from_response_schemas(zero_shot_cot_response_schemas)
zero_shot_cot_format_instructions = zero_shot_cot_output_parser.get_format_instructions()
zero_shot_cot_prompt = PromptTemplate(
    template=zero_shot_cot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": zero_shot_cot_format_instructions},
)
# four-shot cot template.
# - Based on the few-shot COT paper methodology (Wei et al, 2022).
# - Examples sourced from the ECB+ corpus training split:
#   - Two from tps (True Positives) and two from fps (False Positives).
# - Selection was random with a fixed seed of 42 for reproducibility.
four_shot_cot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}
Event1: Lindsay Lohan <m> Checks Into </m> Rehab in Newport Beach ; Investigation of Facility Pending
Event2: Lindsay Lohan has <m> checked into </m> a rehab facility to begin a 90 - day court - ordered stint , handed down as part of a sentence for a 2012 car crash case .
Reasoning: The two events use the same verb, "check into", and main participants, "Lindsay Lohan" and "rehab facility". The two events both highlight Lindsay Lohan's admission into rehab.
Answer: True

Event1:A <m> fire </m> that caused more than $ 1 million in damage to Alaska Gov. Sarah Palin 's church in Wasilla , Alaska , may have been set by an arsonist , investigators say .
Event2: An overnight <m> fire </m> Friday at Alaska Gov. Sarah Palin 's home church caused an estimated $ 1 million in damage , and investigators say it could be the work of an arsonist .
Reasoning: Both events describe a fire at Alaska Gov. Sarah Palin's church in Wasilla, resulting in roughly $1 million in damages. 
Answer: True

Event1; The actress , however , has since <m> checked in </m> to a different rehab but will not face a probation violation for leaving another treatment facility after a few minutes , a prosecutor said Friday .
Event2: Lindsay Lohan <m> Checks Out </m> Of Rehab . . . To Check Into Different Rehab Centre
Reasoning: The first statement details the actress's action of checking into a new rehab facility, while the second specifically highlights Lindsay Lohan's act of checking out of a rehab center. 
Answer: False

Event1: Tara Reid has <m> checked herself into </m> Promises Treatment Center.
Event2: Lindsay Lohan <m> Checks Into </m> Rehab , Checks Out Two Minutes Later.
Reasoning: The events describe the same action of two different individuals.
Answer: False

Event1: {event1}
Event2: {event2}
Your reasoning:
Your answer: 
"""
four_shot_cot_response_schemas = [
    ResponseSchema(
        name="Reasoning",
        description="The reasoning for the answer. Your thinking process."
    ),
    ResponseSchema(
        name="Answer",
        description="The final answer to the question, given the reasoning."
    ),
]
four_shot_cot_output_parser = StructuredOutputParser.from_response_schemas(four_shot_cot_response_schemas)
four_shot_cot_format_instructions = four_shot_cot_output_parser.get_format_instructions()
four_shot_cot_prompt = PromptTemplate(
    template=four_shot_cot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": four_shot_cot_format_instructions},
)