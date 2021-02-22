# Faster RCNN
- tensorpack을 이용한 faster rcnn 구현
- requirements는 faster_rcnn.yml 참고

<br><br>

## 폴더경로 설명
- configs : 모델의 configuration이 저장되는 경로
- custom_data : 데이터셋 저장 경로 (데이터셋 만드는 방법은 1_data_builder.ipynb 참고)
- initial_weight : 초기 wieght 파일 저장경로 
- report : 학습 결과 저장경로
- src : 모델 관련 소스코드

<br>

## 노트북 파일
- **1_data_builder.ipynb** : 학습 가능 한 데이터 형태(COCO type)로 변경 
- **2_config_builder.ipynb** : 모델 학습 config 설정
- **3_training.ipynb** : 모델을 학습하는 script
- **4_eval_pred.ipynb** : 모델의 검증
- **5_model_exporter.ipynb** : serving 가능 한 모델로 export