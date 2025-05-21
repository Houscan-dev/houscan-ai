from rag_chatbot import RAGChatbot, get_available_pdfs


def main():
    try:
        # 사용 가능한 PDF 목록 가져오기
        available_pdfs = get_available_pdfs()
        
        if not available_pdfs:
            print("사용 가능한 PDF 파일이 없습니다.")
            return
            
        print("사용 가능한 PDF 목록:")
        for i, pdf in enumerate(available_pdfs, 1):
            print(f"{i}. {pdf}")
        
        # PDF 선택
        while True:
            try:
                pdf_index = int(input("\n테스트할 PDF 번호를 선택하세요: ")) - 1
                if 0 <= pdf_index < len(available_pdfs):
                    break
                print("유효하지 않은 번호입니다. 다시 선택해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")
        
        selected_pdf = available_pdfs[pdf_index]
        
        # 챗봇 초기화
        print("\n챗봇을 초기화하는 중...")
        chatbot = RAGChatbot(selected_pdf)
        print("챗봇 초기화 완료!")
        
        print("\n챗봇과 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
        
        while True:
            user_input = input("\n질문을 입력하세요: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            try:
                response = chatbot.chat(user_input)
                print("\n답변:", response['answer'])
                if response['sources']:
                    print("\n참고 문서:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"\n{i}. {source['content'][:200]}...")
            except Exception as e:
                print(f"\n오류가 발생했습니다: {str(e)}")
                
    except Exception as e:
        print(f"프로그램 실행 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 