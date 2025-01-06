현재 실행중인 파이썬 파일 찾기
ps -ef | grep python

전체 종료하기
pkill python

특정 ID 종료하기 - 가장 상위 ID를 찾자
kill 1846(해당숫자)


쉘 스크립트 실행하는 방법 정리
<0> 디렉토리 설정
cd /home/work/AHN/dcase2024_task9_baseline

<1> 터미널에 입력
chmod +x run_train.sh

<2>
/home/work/AHN/dcase2024_task9_baseline/run_train.sh
/home/work/AHN/dcase2024_task9_baseline/run_baseline.sh
/home/work/AHN/dcase2024_task9_baseline/run_train_test.sh