#!/bin/bash

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# gunicorn으로 Flask 앱 실행
nohup gunicorn -w 4 -b 127.0.0.1:8000 app:app > gunicorn.log 2>&1 &

# nginx 설정
sudo tee /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80;
    server_name houscan.store;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# nginx 재시작
sudo systemctl restart nginx

echo "설치 및 설정이 완료되었습니다!" 