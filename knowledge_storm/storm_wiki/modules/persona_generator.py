import logging
import re
from typing import Union, List, Optional, Dict

import dspy
import requests
from bs4 import BeautifulSoup


def get_wiki_page_title_and_toc(url):
    """Get the main title and table of contents from an url of a Wikipedia page."""

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the main title from the first h1 tag
    main_title = soup.find("h1").text.replace("[edit]", "").strip().replace("\xa0", " ")

    toc = ""
    levels = []
    excluded_sections = {
        "Contents",
        "See also",
        "Notes",
        "References",
        "External links",
    }

    # Start processing from h2 to exclude the main title from TOC
    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        level = int(
            header.name[1]
        )  # Extract the numeric part of the header tag (e.g., '2' from 'h2')
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        if section_title in excluded_sections:
            continue

        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)

        indentation = "  " * (len(levels) - 1)
        toc += f"{indentation}{section_title}\n"

    return main_title, toc.strip()


class FindRelatedTopic(dspy.Signature):
    """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.
    Please list the urls in separate lines."""

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    related_topics = dspy.OutputField(format=str)


class GenPersona(dspy.Signature):
    """You need to select a group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic. You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of editor 1: description\n2. short summary of editor 2: description\n...
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    examples = dspy.InputField(
        prefix="Wiki page outlines of related topics for inspiration:\n", format=str
    )
    personas = dspy.OutputField(format=str)


class GenNewsPersona(dspy.Signature):
    """You need to generate perspectives for analyzing news coverage of a global event or conflict from different countries' viewpoints. Generate three distinct personas:
    1. A journalist from {country1} analyzing the event/conflict from that country's perspective
    2. A journalist from {country2} analyzing the event/conflict from that country's perspective 
    3. An international media analyst comparing the contrasting viewpoints point by point
    
    Use the custom news sources provided for inspiration on country perspectives and biases.
    If a detailed comparison summary is provided, use it to inform the media analyst's perspective.
    
    Give your answer in the following format: 
    1. short summary of journalist 1: description
    2. short summary of journalist 2: description
    3. short summary of media analyst: description
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    country1 = dspy.InputField(prefix="First country perspective:", format=str, required=False)
    country2 = dspy.InputField(prefix="Second country perspective:", format=str, required=False)
    custom_sources = dspy.InputField(prefix="Custom news sources information:", format=str)
    comparison_summary = dspy.InputField(prefix="Detailed comparison summary (if available):", format=str, required=False)
    personas = dspy.OutputField(format=str)


class CreateWriterWithPersona(dspy.Module):
    """Discover different perspectives of researching the topic by reading Wikipedia pages of related topics."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.gen_news_persona = dspy.ChainOfThought(GenNewsPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None, news_mode: bool = False, custom_sources: List[Dict] = None, explicit_countries: List[str] = None):
        with dspy.settings.context(lm=self.engine):
            if news_mode and custom_sources:
                # Identify countries and find comparison summary if available
                countries = []
                comparison_summary = ""
                
                # Use explicit countries if provided
                if explicit_countries and len(explicit_countries) >= 2:
                    countries = explicit_countries[:2]
                else:
                    # Extract all countries from sources
                    for source in custom_sources:
                        if hasattr(source, 'meta') and 'country' in source.meta:
                            country = source.meta.get('country')
                            if country and country != "International" and country not in countries:
                                countries.append(country)
                
                # Look for comparison summary in custom sources
                for source in custom_sources:
                    if hasattr(source, 'meta') and source.meta.get('type') == 'comparison_summary':
                        comparison_text = "\n".join(source.snippets) if hasattr(source, 'snippets') else ""
                        if comparison_text:
                            comparison_summary = comparison_text
                
                # Ensure we have at least two countries for comparison
                # If we don't have enough, the model will have to infer additional countries
                country1 = countries[0] if len(countries) > 0 else "Unknown Country 1"
                country2 = countries[1] if len(countries) > 1 else "Unknown Country 2"
                
                # Format custom sources for the model
                sources_text = ""
                for i, source in enumerate(custom_sources, 1):
                    if not hasattr(source, 'meta') or source.meta.get('type') != 'comparison_summary':
                        title = source.title if hasattr(source, 'title') else "Unknown Title"
                        url = source.url if hasattr(source, 'url') else f"source-{i}"
                        meta = source.meta if hasattr(source, 'meta') else {}
                        country = meta.get('country', 'Unknown Country')
                        snippets = "\n".join(source.snippets) if hasattr(source, 'snippets') else ""
                        
                        sources_text += f"Source {i} - {title} (from {country}):\n{url}\n{snippets}\n\n"
                
                # Generate personas with explicit country parameters and comparison summary
                gen_persona_output = self.gen_news_persona(
                    topic=topic,
                    country1=country1,
                    country2=country2,
                    custom_sources=sources_text,
                    comparison_summary=comparison_summary
                ).personas
            else:
                # Get section names from wiki pages of relevant topics for inspiration.
                related_topics = self.find_related_topic(topic=topic).related_topics
                urls = []
                for s in related_topics.split("\n"):
                    if "http" in s:
                        urls.append(s[s.find("http") :])
                examples = []
                for url in urls:
                    try:
                        title, toc = get_wiki_page_title_and_toc(url)
                        examples.append(f"Title: {title}\nTable of Contents: {toc}")
                    except Exception as e:
                        logging.error(f"Error occurs when processing {url}: {e}")
                        continue
                if len(examples) == 0:
                    examples.append("N/A")
                gen_persona_output = self.gen_persona(
                    topic=topic, examples="\n----------\n".join(examples)
                ).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        sorted_personas = personas

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=sorted_personas,
            related_topics=related_topics if not news_mode else "News mode - using custom sources",
        )


class StormPersonaGenerator:
    """
    A generator class for creating personas based on a given topic.

    This class uses an underlying engine to generate personas tailored to the specified topic.
    The generator integrates with a `CreateWriterWithPersona` instance to create diverse personas,
    including a default 'Basic fact writer' persona.

    Attributes:
        create_writer_with_persona (CreateWriterWithPersona): An instance responsible for
            generating personas based on the provided engine and topic.

    Args:
        engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The underlying engine used for generating
            personas. It must be an instance of either `dspy.dsp.LM` or `dspy.dsp.HFModel`.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], news_mode: bool = False):
        self.create_writer_with_persona = CreateWriterWithPersona(engine=engine)
        self.news_mode = news_mode

    def generate_persona(self, topic: str, max_num_persona: int = 3, custom_sources: List = None, explicit_countries: List[str] = None) -> List[str]:
        """
        Generates a list of personas based on the provided topic, up to a maximum number specified.

        This method first creates personas using the underlying `create_writer_with_persona` instance
        and then prepends a default 'Basic fact writer' persona to the list before returning it.
        The number of personas returned is limited to `max_num_persona`, excluding the default persona.

        Args:
            topic (str): The topic for which personas are to be generated.
            max_num_persona (int): The maximum number of personas to generate, excluding the
                default 'Basic fact writer' persona.
            custom_sources (List): Optional list of custom news sources to use in news mode.
            explicit_countries (List[str]): Optional list of explicit countries to use for comparison.

        Returns:
            List[str]: A list of persona descriptions, including the default 'Basic fact writer' persona
                and up to `max_num_persona` additional personas generated based on the topic.
        """
        personas = self.create_writer_with_persona(
            topic=topic, 
            news_mode=self.news_mode, 
            custom_sources=custom_sources,
            explicit_countries=explicit_countries
        )
        
        if self.news_mode and custom_sources:
            # The news personas are already created by gen_news_persona
            # We'll return all of them plus the default persona
            default_persona = "Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic."
            considered_personas = [default_persona] + personas.personas
            return considered_personas
        else:
            # Standard STORM behavior
            default_persona = "Basic fact writer: Basic fact writer focusing on broadly covering the basic facts about the topic."
            considered_personas = [default_persona] + personas.personas[:max_num_persona]
            return considered_personas
