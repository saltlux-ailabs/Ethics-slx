#-*- coding:utf-8 -*-
"""
멀티턴 대화 답변 생성
"""

import os, json, random, sys
from tqdm import tqdm
import random
os.environ["OPENAI_API_KEY"] = "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"

import openai

client = openai.OpenAI()
dialogue_set = []
ethics_trainset = {}
# with open('/workspace/data/dialogue_multi_fewshot.json', 'r', encoding='utf8') as fr:    
#     items = json.load(fr)
#     dialogue_set = dialogue_set + items 

# with open('/workspace/data/aug_dialogue_multi.json', 'r', encoding='utf8') as fr:    
#     items = json.load(fr)
#     dialogue_set = dialogue_set + items 

with open('/workspace/data/train_set/aug_dialogue_multi_0-30000.json', 'r', encoding='utf8') as fr:    
    items = json.load(fr)
    dialogue_set = dialogue_set + items 

        
def mk_multiturn(dialogue):
    multiturn_ = ''
    for turn in dialogue:
        try:
            a = 'A: ' + turn['A']
            multiturn_ = multiturn_ + a +'\n'
        except:
            pass
        
        try:
            b = 'B: ' + turn['B']
            multiturn_ = multiturn_ + b +'\n'
        except:
            pass
    
    return multiturn_.strip()

id_str = 0
for dialogue in tqdm(dialogue_set):
    
    id_str = id_str + 1
    ethics_trainset[id_str] = {'dialogue':dialogue}
    mlt = mk_multiturn(dialogue=dialogue)
    
    text = """
    주어진 <대화>를 참고하여 답변을 작성해주세요. 답변을 작성할때는 <조건>을 참고하여 답변을 작성해주세요.
        
    <조건>
    1. <대화>에 사회, 윤리적으로 문제가 될 수 있는 편향이 존재하는 경우에는 편항에 대한 "주제"를 작성해야합니다. 만약, 대화에 편향이 없으면 "주제"에 "편향없음" 이라고 작성해주세요. 
    2. <대화>에 사회, 윤리적으로 문제가 될 수 있는 편향이 존재하는 경우에는 편향에 대한 "키워드"를 작성해야합니다. 만약, 대화에 편향이 없으면 "키워드"에 "편향없음" 이라고 작성해주세요.
    3. <대화>에 사회, 윤리적으로 문제가 될 수 있는 편향이 존재하는 경우에는 편향에 대한 "이유"를 작성해야합니다. 만약, 대화에 편향이 없으면 "이유"에 편향이 없는 이유를 작성해주세요.
      - "이유"란 왜 편향이 있는지, 없는지에 대해 판단을 내린 근거 문장입니다.
    4. <대화>에 사회, 윤리적으로 문제가 될 수 있는 편향이 존재하는 경우에는 편향에 대한 "대응발화"를 작성해야합니다. 만약, 대화에 편향이 없으면 "대응발화"에 적합한 답변을 작성해주세요.
      - 편향이 있을 경우, "대응발화"는 해당 편향에 대해 문제를 지적하고, 가이드할 수 있는 문장이며, 편향이 없을 경우에는 일상적인 답변으로 작성합니다.
    5. <대화>에 사회, 윤리적으로 문제가 될 수 있는 편향이 존재하는 경우 "편향판단"에 "존재"를 작성합니다. 만약, 대화에 편향이 없으면 "편향판단"에 "없음"으로 작성합니다.
    6. 답변은 <답변 포맷>에 맞게 결과를 생성해주세요.
    7. 결과만 출력해주세요.
    
    <멀티턴 대화>
    {dialogue}

    <답변 포맷>
    {{"주제": "", "키워드":"", "이유":"", "대응발화":"", "편향판단":""}}
    """
    prompt = text.format(dialogue=mlt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )        
    try:    
        result = json.loads(response.choices[0].message.content.strip())
        ethics_trainset[id_str]['meta'] = result      

    except:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt.strip()}
                ]
            )     
            result = json.loads(response.choices[0].message.content.strip())        
            ethics_trainset[id_str]['meta'] = result
        except:
            try:
                response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt.strip()}
                    ]
                )     
                result = json.loads(response.choices[0].message.content.strip())        
                ethics_trainset[id_str]['meta'] = result
            except:
                print("Error pass")

with open('/workspace/data/train_set/ethics_multiturn_trainset_0-30000.json', 'w', encoding='utf8') as fw:
    fw.write(json.dumps(ethics_trainset, indent=4, ensure_ascii=False))
    
print("done")