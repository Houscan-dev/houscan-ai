from rag_chatbot import RAGChatbot, get_available_pdfs

def get_user_info():
    """사용자 정보를 입력받는 함수"""
    print("\n사용자 정보를 입력하세요 (입력하지 않으려면 Enter를 누르세요):")
    user_info = {}
    
    fields = {
        'birth_date': '생년월일 (YYYYMMDD)',
        'gender': '성별 (남성/여성)',
        'university_status': '대학생 여부 (재학 중/아니오)',
        'recent_graduate': '최근 졸업생 여부 (예/아니오)',
        'employed': '취업 여부 (예/아니오)',
        'job_seeking': '구직 여부 (예/아니오)',
        'household_type': '가구 유형',
        'parents_own_house': '부모 주택 보유 여부 (예/아니오)',
        'disability_in_family': '가족 중 장애인 여부 (예/아니오)',
        'application_count': '신청 횟수',
        'total_assets': '총 자산 (원)',
        'car_value': '차량 가액 (원)',
        'income_range': '소득 범위',
        'marital_status': '혼인 상태',
        'is_homeless': '노숙인 여부 (예/아니오)'
    }
    
    for key, prompt in fields.items():
        value = input(f"{prompt}: ").strip()
        if value:
            user_info[key] = value
    
    return user_info if user_info else None

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
        
        # 사용자 정보 입력
        user_info = get_user_info()
        
        # 챗봇 초기화
        print("\n챗봇을 초기화하는 중...")
        chatbot = RAGChatbot(selected_pdf, user_info)
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