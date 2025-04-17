import os
import pandas as pd
import gradio as gr
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Tuple, Optional, Any
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상태 정의


class AgentState(TypedDict):
    messages: List
    excel_path: str
    policy_data: pd.DataFrame
    sales_data: pd.DataFrame
    calculation_results: Dict
    current_company: str
    unclear_items: List[str]
    questions: List[str]
    answers: List[str]

# 엑셀 파일 로드 함수


def load_excel_data(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f"엑셀 파일 로드 시작: {excel_path}")

        # 시트 이름 확인
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logger.info(f"엑셀 파일의 시트: {sheet_names}")

        # 시트 이름 확인 및 대소문자 무시
        policy_sheet = next(
            (s for s in sheet_names if s.lower() == "정책"), None)
        sales_sheet = next((s for s in sheet_names if s.lower() == "매출"), None)

        if not policy_sheet:
            raise ValueError("'정책' 시트를 찾을 수 없습니다.")
        if not sales_sheet:
            raise ValueError("'매출' 시트를 찾을 수 없습니다.")

        policy_df = pd.read_excel(excel_path, sheet_name=policy_sheet)
        sales_df = pd.read_excel(excel_path, sheet_name=sales_sheet)

        # 필수 열 확인
        required_policy_columns = ["업체명", "수수료율"]
        required_sales_columns = ["업체명", "매출액"]

        missing_policy_columns = [
            col for col in required_policy_columns if col not in policy_df.columns]
        missing_sales_columns = [
            col for col in required_sales_columns if col not in sales_df.columns]

        if missing_policy_columns:
            raise ValueError(
                f"정책 시트에 필수 열이 없습니다: {', '.join(missing_policy_columns)}")
        if missing_sales_columns:
            raise ValueError(
                f"매출 시트에 필수 열이 없습니다: {', '.join(missing_sales_columns)}")

        # 데이터 타입 확인 및 변환
        try:
            policy_df["수수료율"] = pd.to_numeric(policy_df["수수료율"])
            sales_df["매출액"] = pd.to_numeric(sales_df["매출액"])
        except Exception as e:
            logger.error(f"데이터 타입 변환 중 오류: {str(e)}")
            raise ValueError(f"데이터 타입 변환 중 오류: 수수료율과 매출액은 숫자여야 합니다.")

        return policy_df, sales_df
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"엑셀 파일 로드 중 오류: {str(e)}")
        logger.debug(error_details)
        raise ValueError(f"엑셀 파일 로드 중 오류 발생: {str(e)}")

# 정책 검색 및 적용 함수


def apply_policy(state: AgentState) -> AgentState:
    results = {}
    unclear_items = []

    for company in state["sales_data"]["업체명"].unique():
        company_sales = state["sales_data"][state["sales_data"]
                                            ["업체명"] == company]
        company_policy = state["policy_data"][state["policy_data"]
                                              ["업체명"] == company]

        if company_policy.empty:
            unclear_items.append(f"업체 '{company}'에 대한 정책을 찾을 수 없습니다.")
            continue

        # 기본 정책 정보 추출
        policy_info = company_policy.iloc[0].to_dict()
        commission_rate = policy_info.get("수수료율", 0)

        # 매출액 계산
        total_sales = company_sales["매출액"].sum()
        commission = total_sales * (commission_rate / 100)

        # 추가 정책 적용 (예: 프로모션, 특별 할인 등)
        additional_fees = 0
        for _, policy_row in company_policy.iterrows():
            if "프로모션" in policy_row and policy_row["프로모션"] == "Y":
                additional_fees -= total_sales * 0.02  # 프로모션 할인 2%

            # 불명확한 정책 항목 체크
            for col in policy_row.index:
                if pd.notna(policy_row[col]) and isinstance(policy_row[col], str) and "확인필요" in policy_row[col]:
                    unclear_items.append(
                        f"업체 '{company}'의 '{col}' 항목에 '확인필요' 표시가 있습니다: {policy_row[col]}")

        # 최종 정산금액 계산
        final_amount = total_sales - commission + additional_fees

        results[company] = {
            "매출액": total_sales,
            "수수료율": commission_rate,
            "수수료": commission,
            "추가 조정": additional_fees,
            "최종 정산금액": final_amount
        }

    state["calculation_results"] = results
    state["unclear_items"] = unclear_items

    # 메시지 추가
    if unclear_items:
        state["messages"].append(
            AIMessage(content=f"다음 항목들이 불명확합니다:\n" + "\n".join(unclear_items)))
    else:
        state["messages"].append(AIMessage(content="모든 정책을 명확하게 적용했습니다."))

    return state

# LLM 인스턴스 생성 (재사용)


def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(temperature=temperature)

# 질문 생성 함수


def generate_questions(state: AgentState) -> AgentState:
    if not state["unclear_items"]:
        return state

    llm = get_llm()
    questions = []

    for item in state["unclear_items"]:
        prompt = f"""
        다음은 엑셀 파일의 정책 시트에서 발견된 불명확한 항목입니다:
        {item}
        
        이 항목을 명확히 하기 위해 물어볼 수 있는 구체적인 질문을 1개 생성해주세요.
        질문은 간결하고 명확해야 합니다.
        """

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            question = response.content.strip()
            questions.append(question)
        except Exception as e:
            logger.error(f"질문 생성 중 오류: {str(e)}")
            questions.append(f"[오류] 이 항목에 대한 질문을 생성할 수 없습니다: {item}")

    state["questions"] = questions
    state["messages"].append(AIMessage(
        content="다음 질문들을 통해 불명확한 항목을 명확히 할 수 있습니다:\n" + "\n".join(questions)))

    return state

# 답변 처리 함수


def process_answer(state: AgentState, answer: str) -> AgentState:
    if not state.get("answers"):
        state["answers"] = []

    state["answers"].append(answer)
    state["messages"].append(HumanMessage(content=answer))

    # 답변을 기반으로 정책 업데이트 로직 추가 가능
    llm = get_llm()
    prompt = f"""
    사용자가 다음과 같이 답변했습니다:
    {answer}
    
    이 답변을 바탕으로 정산 계산에 어떤 변경이 필요한지 분석해주세요.
    구체적인 수치나 정책 변경 사항을 명시해주세요.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        analysis = response.content.strip()

        state["messages"].append(AIMessage(content=f"답변 분석 결과:\n{analysis}"))
    except Exception as e:
        logger.error(f"답변 분석 중 오류: {str(e)}")
        state["messages"].append(
            AIMessage(content=f"답변 분석 중 오류가 발생했습니다: {str(e)}"))

    return state

# 결과 요약 함수


def summarize_results(state: AgentState) -> AgentState:
    summary = "## 정산 결과 요약\n\n"

    for company, result in state["calculation_results"].items():
        summary += f"### {company}\n"
        summary += f"- 매출액: {result['매출액']:,.0f}원\n"
        summary += f"- 수수료율: {result['수수료율']}%\n"
        summary += f"- 수수료: {result['수수료']:,.0f}원\n"
        summary += f"- 추가 조정: {result['추가 조정']:,.0f}원\n"
        summary += f"- 최종 정산금액: {result['최종 정산금액']:,.0f}원\n\n"

    if state.get("answers"):
        summary += "## 질의응답 내역\n\n"
        for i, (q, a) in enumerate(zip(state["questions"], state["answers"])):
            summary += f"Q{i+1}: {q}\n"
            summary += f"A{i+1}: {a}\n\n"

    state["messages"].append(AIMessage(content=summary))
    return state

# 라우터 함수


def router(state: AgentState) -> str:
    # 디버깅 정보 출력
    logger.info(
        f"라우터 호출: unclear_items={len(state.get('unclear_items', []))}, questions={len(state.get('questions', []))}, answers={len(state.get('answers', []))}, current_node={state.get('_current_node')}")

    # 명시적 제어 노드가 있으면 해당 노드로 이동
    if state.get("_next_node"):
        next_node = state["_next_node"]
        logger.info(f"라우터 결정: 명시적 제어 - {next_node}")
        # 다음 라우팅에서는 이 값을 사용하지 않도록 삭제
        del state["_next_node"]
        return next_node

    # wait_for_answer 노드에서는 항상 END로 이동하여 스트림 종료
    if state.get("_current_node") == "wait_for_answer":
        logger.info("라우터 결정: END (사용자 입력 대기)")
        return END

    # 계산 결과가 없으면 load_and_apply_policy로 이동
    if not state.get("calculation_results"):
        logger.info("라우터 결정: load_and_apply_policy (계산 결과 없음)")
        return "load_and_apply_policy"
    
    if state.get("unclear_items") and not state.get("questions"):
        logger.info("라우터 결정: generate_questions")
        return "generate_questions"
    elif state.get("questions") and len(state.get("answers", [])) < len(state["questions"]):
        logger.info("라우터 결정: wait_for_answer")
        return "wait_for_answer"
    else:
        logger.info("라우터 결정: summarize")
        return "summarize"

# 챗봇 노드 함수
def chatbot_node(state: AgentState) -> AgentState:
    """
    챗봇 노드 - 사용자 입력에 따라 다음 노드를 결정합니다.
    """
    logger.info("챗봇 노드 실행")
    
    # 현재 상태에 따라 적절한 메시지 생성
    if not state.get("messages"):
        # 초기 메시지
        state["messages"].append(AIMessage(content="엑셀 파일이 로드되었습니다. 정산 계산을 시작합니다."))
    
    # 다음 노드 설정 (항상 load_and_apply_policy로 시작)
    state["_next_node"] = "load_and_apply_policy"
    
    return state

# 그래프 생성


def create_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("load_and_apply_policy", apply_policy)
    workflow.add_node("generate_questions", generate_questions)

    # 사용자 입력 대기 노드 - 상태를 명시적으로 업데이트하고 END로 이동
    def wait_for_answer_fn(state):
        state["_current_node"] = "wait_for_answer"
        # 명시적으로 다음 노드를 END로 설정
        state["_next_node"] = END
        return state

    workflow.add_node("wait_for_answer", wait_for_answer_fn)
    workflow.add_node("summarize", summarize_results)

    # 엣지 추가
    workflow.set_entry_point("chatbot")
    
    # 챗봇에서 모든 노드로 조건부 이동 가능
    workflow.add_conditional_edges(
        "chatbot",
        router,
        {
            "load_and_apply_policy": "load_and_apply_policy",
            "generate_questions": "generate_questions",
            "wait_for_answer": "wait_for_answer",
            "summarize": "summarize",
            END: END
        }
    )
    
    # 다른 노드들의 엣지 설정
    workflow.add_conditional_edges(
        "load_and_apply_policy",
        router,
        {
            "chatbot": "chatbot",
            "generate_questions": "generate_questions",
            "wait_for_answer": "wait_for_answer",
            "summarize": "summarize",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_questions",
        router,
        {
            "chatbot": "chatbot",
            "wait_for_answer": "wait_for_answer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "wait_for_answer",
        router,
        {
            "chatbot": "chatbot",
            "generate_questions": "generate_questions",
            "summarize": "summarize",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "summarize",
        router,
        {
            "chatbot": "chatbot",
            END: END
        }
    )

    # 재귀 제한 증가
    return workflow.compile()

# Gradio 인터페이스


def create_interface():
    graph = create_graph()

    with gr.Blocks(title="엑셀 정산 계산기") as demo:
        gr.Markdown("# 엑셀 정책 기반 정산 계산기")
        gr.Markdown("정책 시트와 매출 시트가 포함된 엑셀 파일을 업로드하세요.")

        with gr.Row():
            excel_file = gr.File(
                label="엑셀 파일 업로드", file_types=[".xlsx", ".xls"])

        chatbot = gr.Chatbot(label="정산 계산 결과", type="messages")
        msg = gr.Textbox(label="질문에 대한 답변 입력")

        def process_file(file):
            if not file:
                return [{"role": "assistant", "content": "엑셀 파일을 업로드해주세요."}], None

            try:
                # 디버깅 메시지 추가
                logger.info(f"파일 처리 시작: {file.name}")

                policy_df, sales_df = load_excel_data(file.name)

                logger.info(f"정책 데이터 로드 완료: {len(policy_df)} 행")
                logger.info(f"매출 데이터 로드 완료: {len(sales_df)} 행")

                # 데이터 요약 로깅
                logger.info(f"업체 목록: {', '.join(policy_df['업체명'].unique())}")
                logger.info(f"총 매출액: {sales_df['매출액'].sum():,.0f}원")

                # 초기 상태 설정
                state = {
                    "messages": [],
                    "excel_path": file.name,
                    "policy_data": policy_df,
                    "sales_data": sales_df,
                    "calculation_results": {},
                    "current_company": "",
                    "unclear_items": [],
                    "questions": [],
                    "answers": []
                }

                # 그래프 실행
                try:
                    # 시작 노드를 chatbot으로 설정
                    events = []
                    try:
                        for event in graph.stream(state):
                            events.append(event)
                            logger.debug(f"이벤트: {event}")
                            if isinstance(event, dict) and event.get("type") == "node":
                                logger.info(f"노드 실행: {event['node']}")
                                if event["node"] == "wait_for_answer":
                                    # wait_for_answer 노드에 도달하면 스트림 종료
                                    logger.info("wait_for_answer 노드 감지, 스트림 종료")
                                    break
                                elif event["node"] == END:
                                    logger.info("END 노드 감지, 스트림 종료")
                                    break
                    except Exception as e:
                        if "'wait_for_answer'" in str(e):
                            # wait_for_answer 관련 오류는 정상적인 종료로 처리
                            logger.info("wait_for_answer 노드에서 종료됨")
                        else:
                            # 다른 오류는 다시 발생시킴
                            raise

                    # 마지막 상태 업데이트
                    if events:
                        for event in reversed(events):
                            if isinstance(event, dict) and event.get("type") == "state":
                                state = event["data"]
                                logger.info("최종 상태 업데이트 완료")
                                break
                except Exception as graph_error:
                    logger.error(f"그래프 실행 오류: {str(graph_error)}")
                    return [{"role": "assistant", "content": f"그래프 실행 중 오류 발생: {str(graph_error)}"}], None

                # 메시지 변환
                chat_messages = []
                for message in state["messages"]:
                    if isinstance(message, HumanMessage):
                        chat_messages.append(
                            {"role": "user", "content": message.content})
                    else:
                        chat_messages.append(
                            {"role": "assistant", "content": message.content})

                if not chat_messages:
                    chat_messages = [
                        {"role": "assistant", "content": "파일이 처리되었지만 메시지가 생성되지 않았습니다."}]

                return chat_messages, state
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"오류 발생: {str(e)}")
                print(error_details)
                return [{"role": "assistant", "content": f"오류 발생: {str(e)}\n\n상세 오류: {error_details}"}], None

        def answer_question(message, state, chatbot):
            if not state or not message:
                return chatbot, state

            try:
                # 답변 처리
                state = process_answer(state, message)

                # 그래프 계속 실행
                try:
                    # 상태에서 _current_node 플래그 제거하여 새로운 실행 시작
                    if "_current_node" in state:
                        del state["_current_node"]
                    
                    # 다음 노드를 generate_questions 또는 summarize로 설정
                    if len(state.get("answers", [])) < len(state.get("questions", [])):
                        state["_next_node"] = "generate_questions"
                    else:
                        state["_next_node"] = "summarize"

                    events = []
                    try:
                        for event in graph.stream(state):
                            events.append(event)
                            if isinstance(event, dict) and event.get("type") == "node":
                                logger.info(f"노드 실행: {event['node']}")
                                if event["node"] == "wait_for_answer" or event["node"] == END:
                                    logger.info(f"{event['node']} 노드 감지, 스트림 종료")
                                    break
                    except Exception as e:
                        if "'wait_for_answer'" in str(e):
                            # wait_for_answer 관련 오류는 정상적인 종료로 처리
                            logger.info("wait_for_answer 노드에서 종료됨")
                        else:
                            # 다른 오류는 다시 발생시킴
                            raise

                    # 마지막 상태 업데이트
                    if events:
                        for event in reversed(events):
                            if isinstance(event, dict) and event.get("type") == "state":
                                state = event["data"]
                                logger.info("최종 상태 업데이트 완료")
                                break
                except Exception as graph_error:
                    logger.error(f"그래프 실행 오류: {str(graph_error)}")
                    state["messages"].append(
                        AIMessage(content=f"처리 중 오류가 발생했습니다: {str(graph_error)}"))
            except Exception as e:
                logger.error(f"답변 처리 중 오류: {str(e)}")
                state["messages"].append(
                    AIMessage(content=f"답변 처리 중 오류가 발생했습니다: {str(e)}"))

            # 메시지 업데이트
            chat_messages = []
            for message in state["messages"]:
                if isinstance(message, HumanMessage):
                    chat_messages.append(
                        {"role": "user", "content": message.content})
                else:
                    chat_messages.append(
                        {"role": "assistant", "content": message.content})

            return chat_messages, state

        # 상태를 저장할 State 컴포넌트 추가
        state = gr.State(None)

        # 이벤트 연결
        excel_file.upload(process_file, excel_file, [chatbot, state])
        msg.submit(answer_question, [msg, state, chatbot], [
                   chatbot, state]).then(lambda: "", None, msg)

    return demo


# 앱 실행
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
