from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

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
        description=("Determines if two marked events in provided sentences refer to the same event."
                     "You answer should only be 'True' or 'False'.")
    ),
]
baseline_output_parser = StructuredOutputParser.from_response_schemas(baseline_response_schemas)
baseline_format_instructions = baseline_output_parser.get_format_instructions()

baseline_prompt = PromptTemplate(
    template=baseline_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": baseline_format_instructions},
)