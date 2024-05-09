## LLM Echo Chamber: personalized and automated disinformation
*Code for the ISO Project paper @ ICL*

<p align="center"><span style="color: red;">!!!warning!!!</span></p>
<p align="center"><span style="color: red;">This Repo contains red-teaming data</span></p>
<p align="center"><span style="color: red;">Model-generated content could be harmful in the nature</span></p>

### Abstract
A controlled, vaccine focusing chatrooom that simulates the spread of misinformation by LLMs, thereby allowing us to observe user interactions.

### A Quick Glance
![](/assets/demo_video.gif)


### Methods
1. Frontend: Streamlit
2. Backend: Prompt Engineering + Finetuning + Langchain + Phi2
3. Deploy: Ngrok


<!-- ### User Interface
![alt text](assets/exp_ss_2.png "user interface") -->

### Paper
the paper could be seen [here](https://drive.google.com/file/d/1K3lsKwyWas0UF3zAjXuzS_wyuyrP70HF/view?usp=drive_link)

### Input Format
* any thing you want but with signs e.g. ./!/?

### deploy public
* streamlit run ./chatbot_frontend.py
* ~/ngrok http 8501

### Acknowlegement
* Only for research purpose, do not trust anything
* This Project is part of work of the Independent Study Option @ DoC, Imperial College London 