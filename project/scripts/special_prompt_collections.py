from langchain import PromptTemplate

# metaphor generation
metaphor_template_single="""
You are a metaphor expert. Your task is to transform specific words in a given sentence into metaphors. These metaphors can only be single-word replacements. Here are the detailed steps you need to follow:

Read the Sentence Provided: Focus on understanding the context and meaning of the sentence.
Review the Word List: This list contains the words you need to transform into metaphors.
Generate Metaphors:
Create 5 distinct single-word metaphors for each word in the list.

Compose a New Sentence:
Replace the original words with your chosen metaphors randomly.
Ensure the new sentence maintains logical and grammatical coherence.
Sentence to Transform:
{sentence}

Word List to Convert into Metaphors:
{trigger_list}

Output Requirements:
Provide your final output in JSON format, including:

The "Original Sentence".
The "Original Word List".
The "Metaphoric Word List" (with your chosen metaphors).
The "Metaphoric Sentence" (the sentence with metaphors incorporated).
Remember, the goal is to use metaphors to convey the original sentence's meaning in a more nuanced or impactful way without altering the core information.
"""
metaphor_prompt_single = PromptTemplate(
    template=metaphor_template_single,
    input_variables=['sentence', 'trigger_list'],
)

metaphor_template_multi="""
You are a metaphor expert. Your task is to transform specific words in a given sentence into metaphors. These metaphors can only be multi-word replacements. Here are the detailed steps you need to follow:

Read the Sentence Provided: Focus on understanding the context and meaning of the sentence.
Review the Word List: This list contains the words you need to transform into metaphors.
Generate Metaphors:
Create 5 distinct multi-word phrasal metaphors for each word in the list.

Compose a New Sentence:
Replace the original words with your chosen metaphors randomly.
Ensure the new sentence maintains logical and grammatical coherence.
Sentence to Transform:
{sentence}

Word List to Convert into Metaphors:
{trigger_list}

Output Requirements:
Provide your final output in JSON format, including:

The "Original Sentence".
The "Original Word List".
The "Metaphoric Word List" (with your chosen metaphors).
The "Metaphoric Sentence" (the sentence with metaphors incorporated).
Remember, the goal is to use metaphors to convey the original sentence's meaning in a more nuanced or impactful way without altering the core information.
"""
metaphor_prompt_multi = PromptTemplate(
    template=metaphor_template_multi,
    input_variables=['sentence', 'trigger_list'],
)