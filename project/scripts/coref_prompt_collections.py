from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# baseline template
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
zeroshot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}
Event1: {event1}
Event2: {event2}
Let's think step by step.
Your reasoning:
Your answer:
"""
cot_response_schemas = [
    ResponseSchema(
        name="Reasoning",
        description="The reasoning for the answer. Your thinking process."
    ),
    ResponseSchema(
        name="Answer",
        description="The final answer to the question, given the reasoning."
    ),
]
cot_output_parser = StructuredOutputParser.from_response_schemas(cot_response_schemas)
cot_format_instructions = cot_output_parser.get_format_instructions()
zeroshot_prompt = PromptTemplate(
    template=zeroshot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": cot_format_instructions},
)
# two shot cot 
twoshot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}

Event1: TRIAL DATE <m> SET </m> FOR MAN ACCUSED OF DOUBLE MURDER IN MILLOM
Event2: PROVISIONAL TRIAL DATE <m> SET </m> FOR CUMBRIA DOUBLE MURDER ACCUSED
Reasoning: Event1 and Event2 both describe the action of setting a trial date, involving the same participant - a man accused of double murder. The events are geographically linked, with Millom located within Cumbria, suggesting a common setting. Additionally, both events focus on a future trial date, highlighting temporal proximity. These parallels in action, participant, location, and timing establish that the events are coreferent and pertain to the same real-world incident.
Answer: True

Event1:`` Tara Reid has <m> checked herself into </m> Promises Treatment Center .
Event2: Lindsay Lohan <m> Checks Into </m> Rehab , Checks Out Two Minutes Later
Reasoning: Despite both events describing someone "checking into" a facility, they are not coreferent due to key differences. Event 1 involves Tara Reid, while Event 2 involves Lindsay Lohan, highlighting different participants. The lack of explicit information about the locations and absence of temporal links further support their distinction. Given these factors, especially the different individuals involved, the events represent separate occurrences despite the similarity in actions.
Answer: False

Event1: {event1}
Event2: {event2}
Your reasoning:
Your answer: 
"""
twoshot_prompt = PromptTemplate(
    template=twoshot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": cot_format_instructions},
)

# four-shot cot template.
fourshot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}

Event1: TRIAL DATE <m> SET </m> FOR MAN ACCUSED OF DOUBLE MURDER IN MILLOM
Event2: PROVISIONAL TRIAL DATE <m> SET </m> FOR CUMBRIA DOUBLE MURDER ACCUSED
Reasoning: Event1 and Event2 both describe the action of setting a trial date, involving the same participant - a man accused of double murder. The events are geographically linked, with Millom located within Cumbria, suggesting a common setting. Additionally, both events focus on a future trial date, highlighting temporal proximity. These parallels in action, participant, location, and timing establish that the events are coreferent and pertain to the same real-world incident.
Answer: True

Event1:`` Tara Reid has <m> checked herself into </m> Promises Treatment Center .
Event2: Lindsay Lohan <m> Checks Into </m> Rehab , Checks Out Two Minutes Later
Reasoning: Despite both events describing someone "checking into" a facility, they are not coreferent due to key differences. Event 1 involves Tara Reid, while Event 2 involves Lindsay Lohan, highlighting different participants. The lack of explicit information about the locations and absence of temporal links further support their distinction. Given these factors, especially the different individuals involved, the events represent separate occurrences despite the similarity in actions.
Answer: False

Event1; Lindsay Lohan <m> Checks Into </m> Rehab in Newport Beach ; Investigation of Facility Pending
Event2: Lindsay Lohan has <m> checked into </m> a rehab facility to begin a 90 - day court - ordered stint , handed down as part of a sentence for a 2012 car crash case .
Reasoning: Two events are considered coreferent due to shared characteristics: both involve Lindsay Lohan checking into a rehabilitation facility, indicating coreference through consistent action. The participant in both events is Lindsay Lohan, reinforcing the likelihood of coreference. The location, Newport Beach, mentioned in the first event, is not contradicted by the second, supporting coreference. Time-wise, both events align with Lindsay's 90-day court-ordered rehab stint following a 2012 car crash, providing a compatible temporal context. These factors - consistent action, participant, non-conflicting location, and temporal alignment - collectively confirm that the events are coreferent, referencing Lindsay Lohan's rehab entry.
Answer: True

Event1: Lindsay Lohan ’ s attorney <m> said </m> Thursday the actress checked into a California rehab facility , but a state official said it is unlicensed to perform the type of restrictive in - patient treatment a judge required the actress to receive in a misdemeanor driving case .
Event2: Santa Monica Chief Deputy City Attorney Terry White <m> confirmed </m> to the The Associated Press that Lohan has checked into an approved rehab facility .
Reasoning: The two events related to Lindsay Lohan's rehab situation are analyzed. Event 1 involves Lindsay Lohan's attorney saying something, while Event 2 features Santa Monica's Chief Deputy City Attorney, Terry White, confirming information. Despite both events focusing on communication about Lohan's situation, they are not coreferent due to distinct actions and participants: one is a general statement and the other an official confirmation, made by a personal attorney and a city attorney, respectively. These differences imply varying perspectives and possibly different timings in the narrative, highlighting separate stages or aspects of the situation.
Answer: False

Event1: {event1}
Event2: {event2}
Your reasoning:
Your answer: 
"""
fourshot_prompt = PromptTemplate(
    template=fourshot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": cot_format_instructions},
)

# eight-shot cot template.
eightshot_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events. Your task is to identify if they both refer to the same event. If they are coreferent, please return "True", otherwise write "False".
{format_instructions}

Event1: TRIAL DATE <m> SET </m> FOR MAN ACCUSED OF DOUBLE MURDER IN MILLOM
Event2: PROVISIONAL TRIAL DATE <m> SET </m> FOR CUMBRIA DOUBLE MURDER ACCUSED
Reasoning: Event1 and Event2 both describe the action of setting a trial date, involving the same participant - a man accused of double murder. The events are geographically linked, with Millom located within Cumbria, suggesting a common setting. Additionally, both events focus on a future trial date, highlighting temporal proximity. These parallels in action, participant, location, and timing establish that the events are coreferent and pertain to the same real-world incident.
Answer: True

Event1:`` Tara Reid has <m> checked herself into </m> Promises Treatment Center .
Event2: Lindsay Lohan <m> Checks Into </m> Rehab , Checks Out Two Minutes Later
Reasoning: Despite both events describing someone "checking into" a facility, they are not coreferent due to key differences. Event 1 involves Tara Reid, while Event 2 involves Lindsay Lohan, highlighting different participants. The lack of explicit information about the locations and absence of temporal links further support their distinction. Given these factors, especially the different individuals involved, the events represent separate occurrences despite the similarity in actions.
Answer: False

Event1; Lindsay Lohan <m> Checks Into </m> Rehab in Newport Beach ; Investigation of Facility Pending
Event2: Lindsay Lohan has <m> checked into </m> a rehab facility to begin a 90 - day court - ordered stint , handed down as part of a sentence for a 2012 car crash case .
Reasoning: Two events are considered coreferent due to shared characteristics: both involve Lindsay Lohan checking into a rehabilitation facility, indicating coreference through consistent action. The participant in both events is Lindsay Lohan, reinforcing the likelihood of coreference. The location, Newport Beach, mentioned in the first event, is not contradicted by the second, supporting coreference. Time-wise, both events align with Lindsay's 90-day court-ordered rehab stint following a 2012 car crash, providing a compatible temporal context. These factors - consistent action, participant, non-conflicting location, and temporal alignment - collectively confirm that the events are coreferent, referencing Lindsay Lohan's rehab entry.
Answer: True

Event1: Lindsay Lohan ’ s attorney <m> said </m> Thursday the actress checked into a California rehab facility , but a state official said it is unlicensed to perform the type of restrictive in - patient treatment a judge required the actress to receive in a misdemeanor driving case .
Event2: Santa Monica Chief Deputy City Attorney Terry White <m> confirmed </m> to the The Associated Press that Lohan has checked into an approved rehab facility .
Reasoning: The two events related to Lindsay Lohan's rehab situation are analyzed. Event 1 involves Lindsay Lohan's attorney saying something, while Event 2 features Santa Monica's Chief Deputy City Attorney, Terry White, confirming information. Despite both events focusing on communication about Lohan's situation, they are not coreferent due to distinct actions and participants: one is a general statement and the other an official confirmation, made by a personal attorney and a city attorney, respectively. These differences imply varying perspectives and possibly different timings in the narrative, highlighting separate stages or aspects of the situation.
Answer: False

Event1: A <m> fire </m> that caused more than $ 1 million in damage to Alaska Gov. Sarah Palin 's church in Wasilla , Alaska , may have been set by an arsonist , 
Event2: An overnight <m> fire </m> Friday at Alaska Gov. Sarah Palin 's home church caused an estimated $ 1 million in damage , and investigators say it could be the work of an arsonist .
Reasoning: Both events revolve around a fire at Sarah Palin's church in Wasilla, Alaska, making the action and location identical. The target of the fire, the church, is a consistent participant in both descriptions. While Event 1 doesn't specify the time, Event 2 describes it as an "overnight fire Friday," without contradicting the first event's timeline. These overlapping details—same action, participant, and location, with compatible temporal information—indicate that the events are coreferent, referring to the same real-world incident, likely an arson at Palin's church.
Answe: True

Event1: In Saturday's semifinals , Creighton ( 25 - 7 ) will play No . 5 seed Indiana State , which <m> defeated </m> Evansville 51 - 50 on Friday .
Event2: Doug McDermott scored 23 points to become Creighton's all - time scoring leader and the top - seeded Bluejays <m> defeated </m> Drake 65 - 53 on Friday in the quarterfinals of the Missouri Valley Conference tournament .
Reasoning: While both events in question describe a team being "defeated," the events are not coreferent due to distinct differences. The teams involved are different: Indiana State defeated Evansville, and Creighton defeated Drake, indicating separate matches. Though likely occurring in the same tournament (Missouri Valley Conference tournament), the importance of location is secondary to the specific matchups. Additionally, the events occurred on different days—Event 1 on a Friday and Event 2 on a different day (Friday in the quarterfinals, contrasting with Saturday's semifinals in Event 1). These factors confirm the events as separate matches within the same tournament.
Answer: False

Event1: Saints RB Pierre Thomas placed on <m> injured reserve </m>
Event2: The Saints have placed running back Pierre Thomas on <m> injured reserve </m> .
Reasoning: Both events consistently describe the action "injured reserve," where Saints running back Pierre Thomas is sidelined from games due to injury. The involvement of the same player in identical roles across both events, along with their setting within the same team context (the Saints), strongly indicates coreference. Additionally, the absence of conflicting details about the timing of these events further reinforces their connection. Therefore, these elements collectively support the conclusion that both events are coreferent, referring to the same incident of Pierre Thomas being placed on injured reserve.
Answer: True

Event1: After skipping out on entering a Newport Beach rehabilitation facility and facing the prospect of arrest for violating her probation , Lindsay Lohan has <m> checked into </m> the Betty Ford Center to begin a 90 - day court - mandated stay in her reckless driving conviction .
Event2: Perennial party girl Tara Reid <m> checked herself into </m> Promises Treatment Center , her rep told People .
Reasoning: These two events involving "checking into" a facility are compared. Although the action is similar, coreference is negated due to differing participants (Lindsay Lohan at the Betty Ford Center vs. Tara Reid at the Promises Treatment Center) and locations. The absence of shared temporal details further supports their distinction. Hence, despite the similarity in action, the events are non-coreferent as they involve different individuals and occur in separate locations, establishing them as independent occurrences.
Answer: False

Event1: {event1}
Event2: {event2}
Your reasoning:
Your answer: 
"""

eightshot_prompt = PromptTemplate(
    template=eightshot_template,
    input_variables=['event1', 'event2'],
    partial_variables={"format_instructions": cot_format_instructions},
)

# ECR explanation
explanation_template = """
You are en expert Event Coreference Resolution System. I will provide two sentences with marked events and the true label of the two events. Your task is to identify why they are coreferent or not coreferent. 
When solving the ECR task, it is important to consider the event actions, participants of actions, locations, and time. 
Event1: {event1}
Event2: {event2}
Gold label: {true_label}
Your answer:
"""

explanation_prompt = PromptTemplate(
    template=explanation_template,
    input_variables=['event1', 'event2', 'true_label'],)