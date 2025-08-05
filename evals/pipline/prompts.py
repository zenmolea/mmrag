

# 检索请求prompt模板
RETRIEVE_REQUEST_PROMPT = """Question: {question}
Options: {options}

To answer the question step by step, you can provide your retrieve request to assist you by the following json format:
{{
    "ASR": Optional[str]. The subtitles of the video that may relavent to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation". 
}}

## Example 1: 
Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
{{
    "ASR": "The location and the color of balloons, the number of the blue balloons.",
    "DET": ["blue ballons", "long table"],
    "TYPE": ["relation", "number"]
}}

## Example 2: 
Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
Your retrieve can be:
{{
    "ASR": null,
    "DET": ["the man in black", "woman"],
    "TYPE": ["location", "relation"]
}}

## Example 3: 
Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
{{
    "ASR": "The country recognized worldwide for its comedy.",
    "DET": null,
    "TYPE": null
}}

Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format."""

# 最终答案prompt模板
FINAL_ANSWER_PROMPT = """Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. 

Question: {question}
Options: {options}

The best answer is:"""

# 视频时间信息模板
VIDEO_TIME_TEMPLATE = "The video lasts for {video_time:.2f} seconds, and {frame_count} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

# 检测信息模板
DETECTION_INFO_TEMPLATE = "Video have {max_frames_num} frames in total, the detected objects' information in specific frames: {detection_info}"

# ASR信息模板
ASR_INFO_TEMPLATE = "Video Automatic Speech Recognition information (given in chronological order of the video): {asr_info}"

# OCR信息模板
OCR_INFO_TEMPLATE = "Video OCR information (given in chronological order of the video): {ocr_info}"

# 帧信息模板
FRAME_INFO_TEMPLATE = "Frame {frame_num}: {frame_info}"

# 场景描述模板
SCENE_DESCRIPTION_TEMPLATE = """Based on the detected objects in the video frame, provide a detailed scene description including:
- Object locations and relationships
- Object counts and properties
- Spatial arrangements
- Temporal context

Detected objects: {objects}
Scene description:"""

# 多模态融合模板
MULTIMODAL_FUSION_TEMPLATE = """Based on the following multimodal information from the video, answer the question:

Visual Information:
{visual_info}

Audio Information:
{audio_info}

Text Information:
{text_info}

Question: {question}
Options: {options}

Please provide the best answer based on all available information."""

# 错误处理模板
ERROR_HANDLING_TEMPLATE = """I encountered an error while processing the video information. 
Please answer the question based on the available information:

Available information: {available_info}
Question: {question}
Options: {options}

The best answer is:"""

# 置信度评估模板
CONFIDENCE_ASSESSMENT_TEMPLATE = """Based on the available information, assess your confidence in answering this question:

Question: {question}
Available information: {available_info}
Information quality: {quality_assessment}

Confidence level (High/Medium/Low): {confidence_level}
Reasoning: {reasoning}

Answer: {answer}"""

# 推理链模板
REASONING_CHAIN_TEMPLATE = """Let's solve this step by step:

1. Question Analysis: {question_analysis}
2. Information Extraction: {info_extraction}
3. Evidence Evaluation: {evidence_evaluation}
4. Logical Reasoning: {logical_reasoning}
5. Answer Selection: {answer_selection}

Final Answer: {final_answer}"""

# 所有prompt模板的字典
PROMPT_TEMPLATES = {
    "retrieve_request": RETRIEVE_REQUEST_PROMPT,
    "final_answer": FINAL_ANSWER_PROMPT,
    "video_time": VIDEO_TIME_TEMPLATE,
    "detection_info": DETECTION_INFO_TEMPLATE,
    "asr_info": ASR_INFO_TEMPLATE,
    "ocr_info": OCR_INFO_TEMPLATE,
    "frame_info": FRAME_INFO_TEMPLATE,
    "scene_description": SCENE_DESCRIPTION_TEMPLATE,
    "multimodal_fusion": MULTIMODAL_FUSION_TEMPLATE,
    "error_handling": ERROR_HANDLING_TEMPLATE,
    "confidence_assessment": CONFIDENCE_ASSESSMENT_TEMPLATE,
    "reasoning_chain": REASONING_CHAIN_TEMPLATE
}

def get_prompt(template_name: str, **kwargs) -> str:
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {template_name}")
    
    template = PROMPT_TEMPLATES[template_name]
    return template.format(**kwargs)

def get_all_prompts() -> dict:
    return PROMPT_TEMPLATES.copy() 