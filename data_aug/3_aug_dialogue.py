"""
멀티턴 대화를 증강
"""

import os, json, random, sys
from tqdm import tqdm
import random
os.environ["OPENAI_API_KEY"] = "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"

import openai

client = openai.OpenAI()
aug_dialogue = []
with open('/workspace/data/train_set/2_dialogue_multi_fewshot.json', 'r', encoding='utf8') as fr:    
    items = json.load(fr)

for i in tqdm(list(range(0,30000))):
    random.shuffle(items)
    dial1 = str(items[0])
    dial2 = str(items[1])
    dial3 = str(items[2])
    text = """
    사회, 윤리적으로 편향이 있을 수 있는 멀티턴 대화를 <조건>에 맞게 생성해주세요. 
        
    <조건>
    1. <주제>에 포함된 편향이 충분히 드러날 수 있는 대화를 생성해주세요.
    2. 대화는 3번 이상 진행되어야 하며 홀수로 구성되어야 합니다.
    3. "A" 화자는 <주제>에 들어있는 편향을 말해야한다.
    4. 대화는 "A" 번째 화자에서 종료되어야 합니다.
    5. 대화가 진행되는 동안 <주제>와 관련된 피드백이나 대응 답변 등을 생성해서는 안됩니다.
    6. <멀티턴 포맷>에 맞게 결과를 생성해주세요.
    7. 결과만 출력해주세요.

    <멀티턴 포맷>
    [
        {{"A":"","B":""}},
        {{"A":"","B":""}},
        {{"A":""}}
    ]
    
    <멀티턴 대화>
    {dial1}
    
    {dial2}
    
    {dial3}
    

    """
    prompt = text.format(dial1=dial1, dial2=dial2, dial3=dial3)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )        
    try:    
        result = json.loads(response.choices[0].message.content.strip())        
        aug_dialogue.append(result)
    except:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt.strip()}
                ]
            )     
            result = json.loads(response.choices[0].message.content.strip())        
            aug_dialogue.append(result)
        except:
            try:
                response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt.strip()}
                    ]
                )     
                result = json.loads(response.choices[0].message.content.strip())        
                aug_dialogue.append(result)
            except:
                pass

    
print(aug_dialogue)
with open('/workspace/data/train_set/aug_dialogue_multi_0-30000.json', 'w', encoding='utf8') as fw:
    fw.write(json.dumps(aug_dialogue, indent=4, ensure_ascii=False))
    
print("done")