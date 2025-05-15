from langchain_core.prompts import ChatPromptTemplate
from langchain_utils import get_llm

def generate_summary(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    """Generate a concise summary of the video content"""
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to generate a summary."
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at summarizing YouTube video content.
        Create a concise summary of the main points and topics covered in the video.
        If video metadata is provided, incorporate that context.
        Keep the summary clear, informative, and well-structured.
        Aim for 3-5 paragraphs that capture the essence of the content."""),
        ("human", """Video details: {video_details}
        
        Please summarize the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate summary
    result = llm.invoke(summary_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content

def extract_key_points(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    """Extract key points from the video content"""
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to extract key points."
    
    key_points_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing YouTube video content.
        Extract the most important key points and insights from the transcript.
        Present these as a bulleted list of 5-10 clear, concise points.
        Focus on the main arguments, conclusions, and takeaways.
        If possible, include approximate timestamps for when key points were mentioned."""),
        ("human", """Video details: {video_details}
        
        Please extract the key points from the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate key points
    result = llm.invoke(key_points_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content

def extract_notable_quotes(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    """Extract notable quotes from the video content"""
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to extract notable quotes."
    
    quotes_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing YouTube video content.
        Extract 5-8 notable, insightful or important quotes from the transcript.
        For each quote:
        1. Include the exact quote in quotation marks
        2. Add a brief explanation of why this quote is significant
        3. Include the approximate timestamp if available
        
        Focus on quotes that capture key insights, memorable statements, or powerful moments."""),
        ("human", """Video details: {video_details}
        
        Please extract notable quotes from the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate quotes
    result = llm.invoke(quotes_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content

def generate_flashcards(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    """Generate study flashcards from the video content"""
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to generate flashcards."
    
    flashcards_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating educational content.
        Create 5-10 high-quality study flashcards based on the video transcript.
        Each flashcard should have:
        1. A clear, concise question that tests understanding of an important concept
        2. A comprehensive yet concise answer that provides the necessary information
        
        Format as:
        Q: [Question]
        A: [Answer]
        
        Focus on key concepts, definitions, and important facts that would be valuable for learning."""),
        ("human", """Video details: {video_details}
        
        Please create study flashcards from the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate flashcards
    result = llm.invoke(flashcards_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content

def generate_concept_map(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    """Generate a text-based concept map of the video content"""
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to generate a concept map."
    
    concept_map_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at knowledge organization and conceptual mapping.
        Create a text-based concept map that shows the relationships between key concepts in the video.
        Structure your response as:
        
        1. Main topic/concept
           ├── Subtopic/concept 1
           │   ├── Related idea 1.1
           │   └── Related idea 1.2
           └── Subtopic/concept 2
               ├── Related idea 2.1
               └── Related idea 2.2
        
        Focus on showing connections and hierarchies between concepts.
        Include 5-10 main concepts with their related ideas and connections."""),
        ("human", """Video details: {video_details}
        
        Please create a concept map from the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate concept map
    result = llm.invoke(concept_map_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content

def generate_practice_questions(docs, video_details=None, llm_model="gpt-3.5-turbo"):
    llm = get_llm(llm_model)
    if not llm:
        return "Please add your OpenAI API key to generate practice questions."
    
    questions_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert educator.
        Create 5 high-quality practice questions based on the video transcript.
        For each question:
        1. Write a clear, specific question that tests understanding of an important concept
        2. Provide 4 multiple-choice options (labeled A, B, C, D)
        3. Indicate the correct answer
        4. Include a brief explanation of why that answer is correct
        
        Create questions that test different levels of understanding, from recall to application and analysis."""),
        ("human", """Video details: {video_details}
        
        Please create practice questions from the following transcript content:
        {text}"""),
    ])
    
    # Combine all document content
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Prepare video details string
    video_details_text = "Not available"
    if video_details and "error" not in video_details:
        video_details_text = f"Title: {video_details['title']}\nAuthor: {video_details['author']}"
    
    # Generate practice questions
    result = llm.invoke(questions_prompt.format(
        text=full_text,
        video_details=video_details_text
    ))
    
    return result.content