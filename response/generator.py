from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.settings import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_SEED

def generate_response(query, context_text):
    """컨텍스트를 바탕으로 응답 생성"""
    # 간단한 템플릿 사용
    chat = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED  # 고정된 시드 값 추가
    )

    system_template = """
    당신은 공공입찰 공고 정보를 제공하는 시스템입니다.
    사용자 질문에 대해 제공된 메타데이터 정보를 바탕으로 간결하고 정확한 답변을 제공하세요.
    메타데이터에 포함된 내용만을 바탕으로 답변하며, 검색된 메타데이터가 없으면 그렇게 알려주세요.

    제공된 메타데이터:
    {context}
    """

    human_template = "{question}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    response = chat.invoke(
        chat_prompt.format_messages(
            context=context_text,
            question=query
        )
    )

    return response.content