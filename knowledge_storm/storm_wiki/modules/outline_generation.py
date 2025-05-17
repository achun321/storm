from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    The interface for outline generation stage. Given topic, collected information from knowledge
    curation stage, generate outline for the article.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], news_mode: bool = False):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm, news_mode=news_mode)
        self.news_mode = news_mode

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
        explicit_countries: Optional[list] = None,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Generates an outline for an article based on the specified topic and the information
        gathered during the knowledge curation stage. This method can optionally return both the
        final article outline and a draft outline if required.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            old_outline (Optional[StormArticle]): An optional previous version of the article outline that can
                be used for reference or comparison. Defaults to None.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the outline generation process, such as when the information
                organization starts. Defaults to None.
            return_draft_outline (bool): A flag indicating whether the method should return both the final article
                outline and a draft version of the outline. If False, only the final article outline is returned.
                Defaults to False.
            explicit_countries (Optional[list]): A list of countries to use in news mode for country-specific sections.

        Returns:
            Union[StormArticle, Tuple[StormArticle, StormArticle]]: Depending on the value of `return_draft_outline`,
                this method returns either a single `StormArticle` object containing the final outline or a tuple of
                two  `StormArticle` objects, the first containing the final outline and the second containing the
                draft outline.
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        
        # Initialize country data and comparison summary
        comparison_summary = None
        country1 = None
        country2 = None
        
        if self.news_mode and explicit_countries and len(explicit_countries) >= 2:
            country1 = explicit_countries[0]
            country2 = explicit_countries[1]
            
            # Look for comparison summary
            for url, info in information_table.url_to_info.items():
                if hasattr(info, 'meta') and info.meta.get('type') == 'comparison_summary':
                    comparison_summary = "\n".join(info.snippets) if hasattr(info, 'snippets') else ""
                    break
        
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
            country1=country1,
            country2=country2,
            comparison_summary=comparison_summary,
        )
        article_with_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        article_with_draft_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page or news article."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], news_mode: bool = False):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine
        self.news_mode = news_mode

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
        country1: Optional[str] = None,
        country2: Optional[str] = None,
        comparison_summary: Optional[str] = None,
    ):
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        conv = ArticleTextProcessing.remove_citations(conv)
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 5000)

        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                if self.news_mode and country1 and country2:
                    # For news mode, we'll directly create a country-based outline
                    news_outline = f"""# Introduction
A concise overview of the {topic} and why it's significant.

# Background
Historical context and key events leading up to the current situation regarding {topic}.

# {country1}'s Perspective
## Official Statements and Policies
## Media Coverage and Public Opinion
## Key Actions and Reactions

# {country2}'s Perspective
## Official Statements and Policies
## Media Coverage and Public Opinion
## Key Actions and Reactions

# Comparative Analysis
## Areas of Agreement
## Points of Contention
## Implications for Bilateral Relations

# Recent Developments

# International Response

# Conclusion
"""
                    old_outline = ArticleTextProcessing.clean_up_outline(news_outline)
                else:
                    # Use regular Wikipedia outline format
                    old_outline = ArticleTextProcessing.clean_up_outline(
                        self.draft_page_outline(topic=topic).outline
                    )
                
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(
                        outline=old_outline
                    )
            
            # Further refine the outline with conversation info
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(
                    topic=topic, old_outline=old_outline, conv=conv, country1=country1, country2=country2, comparison_summary=comparison_summary
                ).outline
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        return dspy.Prediction(outline=outline, old_outline=old_outline)


class WritePageOutline(dspy.Signature):
    """Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)


class NaiveOutlineGen(dspy.Module):
    """Generate the outline with LLM's parametric knowledge directly."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)


class WritePageOutlineFromConv(dspy.Signature):
    """
    When country1 and country2 are provided (news mode):
    Refine the news article outline comparing viewpoints from different countries based on the conversation history.
    The structure should include:
    1. Country-specific sections for {country1} and {country2} with their perspectives, policies, and reactions
    2. A comparative analysis section highlighting agreements, differences, and implications
    3. Background information and international responses
    4. Use "#" for section titles, "##" for subsection titles
    
    For regular Wikipedia mode (no countries specified):
    Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the information learned from an information-seeking conversation to make it more informative.
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    old_outline = dspy.InputField(prefix="Current outline:\n", format=str)
    country1 = dspy.InputField(prefix="First country perspective (if news mode):", format=str, required=False)
    country2 = dspy.InputField(prefix="Second country perspective (if news mode):", format=str, required=False)
    comparison_summary = dspy.InputField(prefix="Comparison summary between countries (if news mode):", format=str, required=False)
    outline = dspy.OutputField(
        prefix='Write the improved outline (Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, ...):\n',
        format=str,
    )

    def forward(self, topic, conv, old_outline, country1=None, country2=None, comparison_summary=None):
        # If in news mode with countries specified, use a structured news outline
        if country1 and country2:
            # Create a structured news article outline with specific country perspectives
            # Include any comparison summary info in the structure
            comparison_context = ""
            if comparison_summary:
                comparison_context = f"\nContext from analysis: {comparison_summary}\n"
                
            news_outline = f"""# Introduction
A concise overview of the {topic} and why it's significant in the context of {country1}-{country2} relations.

# Background
Historical context and key events leading up to the current situation regarding {topic}.

# {country1}'s Perspective
## Official Statements and Policies
The official position of {country1}'s government on {topic}.
## Media Coverage and Public Opinion
How {country1}'s media and public view the {topic}.
## Key Actions and Reactions
Specific actions taken by {country1} related to {topic}.

# {country2}'s Perspective
## Official Statements and Policies
The official position of {country2}'s government on {topic}.
## Media Coverage and Public Opinion
How {country2}'s media and public view the {topic}.
## Key Actions and Reactions
Specific actions taken by {country2} related to {topic}.

# Comparative Analysis{comparison_context}
## Areas of Agreement
Points where {country1} and {country2} share common ground on {topic}.
## Points of Contention
Key differences in the positions of {country1} and {country2}.
## Implications for Bilateral Relations
How the {topic} affects relations between {country1} and {country2}.

# Recent Developments
Latest events and changes in the situation regarding {topic}.

# Conclusion
Summary of key points and potential future directions for {topic} in {country1}-{country2} relations.
"""
            return {"outline": news_outline.strip()}
        else:
            # Use traditional processing for non-news mode
            return dspy.Predict.__call__(self, topic=topic, conv=conv, old_outline=old_outline)
