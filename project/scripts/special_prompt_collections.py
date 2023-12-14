from langchain import PromptTemplate

# Davidson Strategy
davidson_template="""
You are an expert in Donald Davidson's philosophy about event identity. Your task is the identify if the given two events (Event1, Event2) are coreferent.
Hint: First identify the event triggers marked between <m> and </m> of the two events. Focussing on the event triggers, next, apply the cause and effect theory of events to check if the two events have the same cause and effect. If they do, your Answer should be True otherwise, False.
Make sure your output is in JSON format as follows:
{{
  "Event1 Trigger": Text corresponding to Event 1's Trigger between <m> and </m>
  "Event2 Trigger": Text corresponding to Event 2's Trigger between <m> and </m>
  "Cause-effect Explanation": Explanation for the same cause and effect of Event1 and Event2
   "Answer": True if Event1 is identical to Event 2 or False if they are not
}}

Now predict if the following two event are coreferent:

Event1: {event1}
Event2: {event2}
"""
davidson_prompt = PromptTemplate(
    template=davidson_template,
    input_variables=['event1', 'event2'],
)

# Quine Strategy
quine_template="""
You are an expert in Quine's theory about event identity. Your task is the identify if the given two events (Event1, Event2) are coreferent.
Hint: First identify the event triggers marked between <m> and </m> of the two events. Focussing on the event triggers, next, identify if both events occupy the same Location and Time and are involving the same participants. If they do, your Answer should be True otherwise, False.
Make sure your output is in JSON format as follows:
{{
  "Event1 Trigger": Text corresponding to Event 1's Trigger between <m> and </m>
  "Event2 Trigger": Text corresponding to Event 2's Trigger between <m> and </m>
  "Spatio-temporal Explanation": Summary of the spatiotemporal relations between Event1 and Event2
   "Answer": True if Event1 is identical to Event 2 or False if they are not
}}

Now predict for the following two events:

Event1: {event1}
Event2: {event2}
"""
quine_prompt = PromptTemplate(
    template=quine_template,
    input_variables=['event1', 'event2'],
)

# AMR Strategy
amr_template="""
You are an expert in Abstract Meaning Representation (AMR). Your task is the identify if the given two events (Event1, Event2) are coreferent.
Hint: First identify the event triggers marked between <m> and </m> of the two events. Focussing on the event triggers, next, generate the AMR graphs of the two events. Comparing the two graphs, make a judgement of coreference. If they are, your Answer should be True otherwise, False.

Be extremely precise in the prediction.

Make sure your output is in JSON format as follows::
{{
  "Event1 Trigger": Text corresponding to Event 1's Trigger between <m> and </m>
  "Event2 Trigger": Text corresponding to Event 2's Trigger between <m> and </m>
  "AMR-likeness Explanation": Summary of the AMR graph likeness between Event1 and Event2
   "Answer": True if Event1 is identical to Event 2 or False if they are not
}}

Now predict for the following two events:

Event1: {event1}
Event2: {event2}
"""
amr_prompt = PromptTemplate(
    template=amr_template,
    input_variables=['event1', 'event2'],
)

# ARG
arg_template="""
You are an expert in Event Argument Extraction. Your task is the identify if the given two events (Event1, Event2) are coreferent.
Hint: First identify the event triggers marked between <m> and </m> of the two events. Focussing on the triggers, check if they have compatible event types. Then, extract ARG-0 (agent), ARG-1 (patient or theme), ARG-Location (Location), and ARG-Time. Comparing the arguments of the two events, make a judgment of coreference. Some arguments will be missing in the text. Make your best guess if they are coreferent. If they are, your Answer should be True otherwise, False.

Make sure your output is in JSON format as follows::
{{
  "Event1 Trigger": Text corresponding to Event 1's Trigger between <m> and </m>
  "Event2 Trigger": Text corresponding to Event 2's Trigger between <m> and </m>
  "Argument Explanation": Summary of the Arguments likeness between Event1 and Event2
   "Answer": True if Event1 is identical to Event 2 or False if they are not
}}

Now predict for the following two events:

Event1: {event1}
Event2: {event2}
"""
arg_prompt = PromptTemplate(
    template=arg_template,
    input_variables=['event1', 'event2'],
)

# Adversarial
adv_template="""
Paraphrase this Sentence from a news article by using figurative language to describe the events in the Sentence:

Make sure you change the event triggers.

Generate 3 samples ordered by the esoteric nature in ascending order in JSON Formate as follows:
{{
   "Less Esoteric" :  "Least Esoteric paraphrasing",
   "Moderately Esoteric" :  "Moderately Esoteric paraphrasing",
   "Most Esoteric" :  "Most Esoteric paraphrasing"
}}

Sentence: {sentence}
"""
adv_prompt = PromptTemplate(
    template=adv_template,
    input_variables=['sentence'],
)

adv_template_v2="""
You are an expert Paraphraser.

Paraphrase this sentence from a news article by using figurative language to describe the events in the Sentence:

I will provide the Sentence and the Event Triggers with it. 

First, make sure you change the event triggers in the sentences. Then identify the Named Entities in the sentences. Finally, use this information for generation.

Generate 4 samples ordered by the esoteric nature in ascending order in JSON Format as follows:
{{
   "Simple Language": "Paraphrasing using simplification of words"
   "Less Esoteric" :  "Least Esoteric paraphrasing",
   "Moderately Esoteric" :  "Moderately Esoteric paraphrasing",
   "Most Esoteric" :  "Most Esoteric paraphrasing"
}}

Event Triggers: {mention_texts}
Sentence: {sentence}
"""
adv_prompt_v2 = PromptTemplate(
    template=adv_template_v2,
    input_variables=['mention_texts', 'sentence'],
)


# Tagging
tag_template="""
You are an expert Event Trigger Tagger.

Original Marked Sentence: {original_sentence}

Original Event Trigger: {original_event_trigger}

Paraphrased Sentence: {paraphrased_sentence}

Please find the Paraphrased Trigger in the Paraphrased Sentence. Then, mark Paraphrased Trigger in Paraphrased Sentence with <m> and </m>
Your output should be in JSON Format as follows:
{{
   Paraphrased Trigger: 
   Paraphrased Marked Sentence:
}}
Be concise
"""
tag_prompt = PromptTemplate(
    template=tag_template,
    input_variables=['original_sentence', 'original_event_trigger', 'paraphrased_sentence'],
)
