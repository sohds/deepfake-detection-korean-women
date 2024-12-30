# deepfake-detection-korean-women
## Grad-CAM++와 Activation Map을 활용한 한국인 여성 중심 딥페이크 탐지 특징 분석

- 2024-2 서울여자대학교 데이터사이언스학과 데이터사이언스캡스톤디자인2 4조
### 👥 Members
<div align="center">
<br>
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kyuriya">
        <img src="./readme_photos/profile/kyuriya.png" width="200px;" alt=""/>
        <br />
        <sub>김규리</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 4학년</sub>
        <br>
        <sub>kyureekim@swu.ac.kr</sub>
    </td>
    <td align="center">
      <a href="https://github.com/seiinkiim">
        <img src="./readme_photos/profile/seiinkiim.png" width="200px;" alt=""/>
        <br />
        <sub>김세인</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 4학년</sub>
        <br>
        <sub>ssen3174@swu.ac.kr</sub>
    </td>
    <td align="center">
      <a href="https://github.com/sohds">
        <img src="./readme_photos/profile/sohds.png" width="200px;" alt=""/>
        <br />
        <sub>오서연</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 3학년</sub>
        <br>
        <sub>osy.seoyeon5@gmail.com</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/stxllaaa">
        <img src="./readme_photos/profile/stxllaaa.png" width="200px;" alt=""/>
        <br />
        <sub>조유라</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 4학년</sub>
        <br>
        <sub>stxllaaa03@gmail.com</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Choi-Daye">
        <img src="./readme_photos/profile/Choi-Daye.png" width="200px;" alt=""/>
        <br />
        <sub>최다예</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 4학년</sub>
        <br>
        <sub>allyes1227@swu.ac.kr</sub>
    </td>
    <td align="center">
      <a href="https://github.com/ejrdn">
        <img src="./readme_photos/profile/ejrdn.png" width="200px;" alt=""/>
        <br />
        <sub>최덕우</sub>
      </a>
        <br>
        <sub>서울여대 데이터사이언스학과 4학년</sub>
        <br>
        <sub>ejrdn1019@swu.ac.kr</sub>
    </td>
  </tr>
</table>
</div>
*Every Member has equal contribution.
<br>

## 📥 Dataset
1. <a href="https://deepbrainai-research.github.io/kodf/">KoDF: A Large-scale Korean DeepFake Detection Dataset </a>
2. <a href="https://liming-jiang.com/projects/DrF1/DrF1.html">DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection</a>
3. <a href="https://ieeexplore.ieee.org/document/4404053">The CAS-PEAL large-scale Chinese face database and baseline evaluations</a>
4. <a href="https://paperswithcode.com/dataset/jaffe">The Japanese Female Facial Expression (JAFFE) Dataset</a>

## 💡 Research Goal
- 딥페이크 탐지에 있어서, 인종별 정확도와 성별별로의 정확도 차이가 확연히 존재함.
  - 특히, 아시안과 여성이 있어서 탐지 정확도가 다른 비교군에 비교해 낮음을 확인함.
- 하지만, 딥페이크 피해 중 여성과 남성의 비율은 99:1이며 이 중 피해 국가는 대한민국이 약 57%로 가장 큰 피해국가로 꼽힘.
- 딥페이크 탐지 모델별 각 인종별 및 동양 국가별 여성에 대해 어떤 부분을 중심으로 집중하는지 확인해보고자 함.

<br>
<br>

---
- 2025 HCI Korea 투고, 논문 Accept 시 README 및 코드 정리 예정