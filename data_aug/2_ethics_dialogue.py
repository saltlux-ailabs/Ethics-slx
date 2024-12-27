"""
주제 문장을 멀티턴 대화 형태로 변환
"""

import os, json, random, sys
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "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"

import openai

client = openai.OpenAI()
dialogue_fewsht = []
with open('/workspace/data/multi_fewshot_sent.json', 'r', encoding='utf8') as fr:    
    items = json.load(fr)

for item in tqdm(items):  
    topic = item['sentence']
    text = """
    주어진 <주제>와 관련있는 멀티턴 대화를 <조건>에 맞게 생성해주세요.
    생성된 멀티턴 대화는 <주제>에 편향이 존재할 경우, 해당 내용이 충분히 반영되어야 하며, 편향에 대한 해결책을 절대 줘서는 안됩니다.
    생성된 멀티턴 대화는 <멀티턴 포맷>에 맞게 작성해주세요.
    
    <주제>
    {topic}
    
    <조건>
    1. <주제>에 포함된 편향이 충분히 드러날 수 있는 대화를 생성해주세요.
    2. 대화는 3번 이상 진행되어야 하며 홀수로 구성되어야 합니다.
    3. "A" 화자는 <주제>에 들어있는 편향을 말해야한다.
    4. 대화는 "A" 번째 화자에서 종료되어야 합니다.
    5. 대화가 진행되는 동안 <주제>와 관련된 피드백이나 대응 답변 등을 생성해서는 안됩니다.
    6. 결과만 출력해주세요.
    
    <멀티턴 포맷>
    [
        {{"A":"","B":""}},
        {{"A":"","B":""}},
        {{"A":""}}
    ]
    """
    prompt = text.format(topic=topic)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )        
    try:    
        result = json.loads(response.choices[0].message.content.strip())        
        dialogue_fewsht.append(result)
    except:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt.strip()}
                ]
            )     
            result = json.loads(response.choices[0].message.content.strip())        
            dialogue_fewsht.append(result)
        except:
            try:
                response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt.strip()}
                    ]
                )     
                result = json.loads(response.choices[0].message.content.strip())        
                dialogue_fewsht.append(result)
            except:
                pass

    
print(dialogue_fewsht)
with open('/workspace/data/dialogue_multi_fewshot.json', 'w', encoding='utf8') as fw:
    fw.write(json.dumps(dialogue_fewsht, indent=4, ensure_ascii=False))
    
print("done")